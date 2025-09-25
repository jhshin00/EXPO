from functools import partial
from typing import Dict, Optional, Sequence, Tuple
import logging

import numpy as np
import flax
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from flax import struct
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training.train_state import TrainState

from expo_pi0.utils.pi0_utils import TrainStatePi0
from expo_pi0.agents.agent import Agent
from expo_pi0.agents.temperature import Temperature
from expo_pi0.distributions import TanhNormal
from expo_pi0.networks import (
    MultiEncoder,
    MultiStateActionValue,
    Ensemble,
    subsample_ensemble,
)

from openpi.models import pi0 as _pi0
from openpi.models import model as _model

def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):

    q_values = critic_fn({'params': critic_params}, observations, actions)
    if q_values.ndim == 2:
        q_values = q_values.min(axis=0)
    return q_values


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations, actions) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, actions)
    return dist.sample(seed=key)

def _make_critic_step(critic_apply_fn, target_apply_fn, critic_tx):
    @partial(jax.jit, static_argnames=("discount","tau"))
    def step(critic_params, critic_opt_state, target_params, rng, *,
             observations, actions, rewards, next_observations, masks, next_actions,
             discount: float, tau: float):

        rng, dropout_key = jax.random.split(rng)

        H = actions.shape[1]
        gamma_vec = jnp.power(discount, jnp.arange(H))[None, :]
        G_n = jnp.sum(gamma_vec * rewards, axis=1)

        next_qs = target_apply_fn(
            {"params": target_params},
            next_observations,
            next_actions,
            True,
            rngs={"dropout": dropout_key},
        )
        next_q = next_qs.min(axis=0) if next_qs.ndim == 2 else next_qs
        target_q = jax.lax.stop_gradient(G_n + (discount**H) * masks.astype(jnp.float32) * next_q)

        def loss_fn(p):
            qs = critic_apply_fn(
                {"params": p},
                observations,
                actions,
                True,
                rngs={"dropout": dropout_key},
            )  # train=True with rng
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(loss_fn, has_aux=True)(critic_params)
        updates, new_opt_state = critic_tx.update(grads, critic_opt_state, critic_params)
        new_params = optax.apply_updates(critic_params, updates)
        new_target_params = optax.incremental_update(new_params, target_params, tau)
        return new_params, new_opt_state, new_target_params, info
    return step


def _make_edit_step(edit_apply_fn, critic_apply_fn, temp_apply_fn, edit_tx):
    @partial(jax.jit, static_argnames=("edit_scale", "entropy_scale"))
    def step(edit_params, edit_opt_state, rng, *,
             obs, act, edit_scale, entropy_scale,
             critic_params, temp_params):
        rng, sample_rng, dropout_key = jax.random.split(rng, 3)

        def loss_fn(p):
            dist = edit_apply_fn({"params": p}, obs, act, training=True, rngs={"dropout": dropout_key})
            edit_flat = dist.sample(seed=sample_rng)       # [B, H*A]
            logp = dist.log_prob(edit_flat)         # [B]

            d = edit_flat.shape[-1]
            logp = logp - d * jnp.log(edit_scale + 1e-8)
            edit_flat = edit_flat * edit_scale
            edit = edit_flat.reshape(act.shape)
            actions = act + edit

            qs = critic_apply_fn({"params": critic_params}, obs, actions, True, rngs={"dropout": dropout_key})
            q = qs.mean()

            alpha = temp_apply_fn({"params": temp_params})
            loss = (entropy_scale * logp * jax.lax.stop_gradient(alpha) - jax.lax.stop_gradient(q)).mean()
            info = {
                "edit_q": q,
                "edit_actor_loss": loss,
                "entropy": (-logp).mean(),
                "logp_mean": logp.mean(),
            }
            return loss, info

        grads, info = jax.grad(loss_fn, has_aux=True)(edit_params)
        updates, new_opt = edit_tx.update(grads, edit_opt_state, edit_params)
        new_params = optax.apply_updates(edit_params, updates)
        return new_params, new_opt, info
    return step


def _make_temp_step(temp_apply_fn, temp_tx):
    @jax.jit
    def step(temp_params, temp_opt_state, *, entropy, target_entropy):
        def loss_fn(p):
            alpha = temp_apply_fn({"params": p})
            loss = alpha * (entropy - target_entropy).mean()
            return loss, {"alpha": alpha, "entropy": entropy, "target_entropy": target_entropy}
        grads, info = jax.grad(loss_fn, has_aux=True)(temp_params)
        updates, new_opt = temp_tx.update(grads, temp_opt_state, temp_params)
        new_params = optax.apply_updates(temp_params, updates)
        return new_params, new_opt, info
    return step


class EXPOPi0Learner(Agent):
    critic: TrainState
    target_critic: TrainState
    actor: TrainStatePi0
    edit_actor: TrainState
    temp: TrainState

    action_dim: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    max_token_len: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_action_scale: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    entropy_scale: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    actor_drop: Optional[float] = struct.field(pytree_node=False)
    use_critic_pi0: bool = struct.field(pytree_node=False)
    img_dim: int = struct.field(pytree_node=False)
    txt_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    vocab_size: int = struct.field(pytree_node=False)
    image_keys: Optional[Sequence[str]] = struct.field(pytree_node=False)
    d_model: int = struct.field(pytree_node=False)
    n_layers: int = struct.field(pytree_node=False)
    kernel_size: int = struct.field(pytree_node=False)
    resume: bool = struct.field(pytree_node=False)
    overwrite: bool = struct.field(pytree_node=False)
    params_path: Optional[str] = struct.field(pytree_node=False)

    _critic_step: any = struct.field(pytree_node=False, default=None)
    _edit_step: any = struct.field(pytree_node=False, default=None)
    _temp_step: any = struct.field(pytree_node=False, default=None)
        
    @classmethod
    def create(
        cls,
        seed: int,
        actor_lr: float = 2.5e-4,
        edit_actor_lr: float = 2.5e-4,
        critic_lr: float = 1e-4,
        temp_lr: float = 1e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0, 
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        adjust_target_entropy: bool = False, 
        N: int = 1,
        T: int = 10,
        n_edit_samples: int = 0,
        edit_action_scale: float = 1.0, 
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = 48,
        actor_drop: Optional[float] = None,
        use_critic_pi0: bool = False,
        img_dim: int = 512,
        txt_dim: int = 512,
        state_dim: int = 256,
        vocab_size: int = 257152,
        image_keys: Optional[Sequence[str]] = None,
        d_model: int = 256,
        n_layers: int = 2,
        kernel_size: int = 3,
        resume: bool = False,
        overwrite: bool = True,
        params_path: Optional[str] = None,
    ):
        if target_entropy is None:
            target_entropy = -action_dim*action_horizon / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_config = _pi0.Pi0Config(
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

        actor = TrainStatePi0.create(
            pi0_config=actor_config,
            rng=actor_key,
            overwrite=overwrite,
            resume=resume,
            params_path=params_path,
        )

        observations = actor_config.fake_obs()
        actions = actor_config.fake_act()

        edit_actor_base_cls = partial(
            MultiEncoder,
            img_dim=img_dim,
            txt_dim=txt_dim,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            use_image=True,
            use_text=True,
            use_state=True,
            vocab_size=vocab_size,
            image_keys=image_keys,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout_rate=actor_drop,
            activate_final=True,
            use_pnorm=use_pnorm,
            use_layer_norm=False,
        )

        edit_actor_def = TanhNormal(edit_actor_base_cls, action_horizon*action_dim) # [H*A] output
        edit_actor_params = edit_actor_def.init(actor_key, observations, actions)["params"]
        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply,
            params=edit_actor_params,
            tx=optax.adam(learning_rate=edit_actor_lr),
        )

        critic_base_cls = partial(
            MultiEncoder,
            img_dim=img_dim,
            txt_dim=txt_dim,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            use_image=True,
            use_text=True,
            use_state=True,
            vocab_size=vocab_size,
            image_keys=image_keys,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout_rate=actor_drop,
            activate_final=True,
            use_pnorm=use_pnorm,
            use_layer_norm=False,
        )
        critic_cls = partial(MultiStateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)

        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        learner = cls(
            rng=rng,
            actor=actor,
            edit_actor=edit_actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale, 
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            edit_action_scale=edit_action_scale,
            N=N,
            n_edit_samples=n_edit_samples,
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
            T=T,
            actor_drop=actor_drop,
            use_critic_pi0=use_critic_pi0,
            img_dim=img_dim,
            txt_dim=txt_dim,
            state_dim=state_dim,
            vocab_size=vocab_size,
            image_keys=image_keys,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            overwrite=overwrite,
            params_path=params_path,
            resume=resume,
        )

        return learner.replace(
            _critic_step=_make_critic_step(learner.critic.apply_fn, learner.target_critic.apply_fn, learner.critic.tx),
            _edit_step=_make_edit_step(learner.edit_actor.apply_fn, learner.critic.apply_fn, learner.temp.apply_fn, learner.edit_actor.tx),
            _temp_step=_make_temp_step(learner.temp.apply_fn, learner.temp.tx),
        )

    def eval_actions(self, observations: _model.Observation) -> _model.Actions:
        if self.N > 1:
            # Multiple samples case
            rngs = jax.random.split(self.rng, self.N + 2)  # +2 for critic and edit samples
            rng, sample_rngs, edit_rngs = rngs[0], rngs[1:self.N+1], rngs[self.N+1:]
            
            # Generate N action samples
            action_samples = jax.vmap(
                lambda rng: self.actor.sample_actions(rng, observations, num_steps=self.T)
            )(sample_rngs)  # [N, H, A]
            
            if self.n_edit_samples > 0:
                anchor_actions = action_samples[:self.n_edit_samples]  # [n, H, A]
                
                def edit_single_action(rng, anchor_action):
                    # Add batch dimension to anchor_action: [H, A] -> [1, H, A]
                    anchor_action_batched = anchor_action[None, :, :]
                    edit_action = _sample_actions(rng, self.edit_actor.apply_fn, self.edit_actor.params, observations, anchor_action_batched)
                    # edit_action is [1, H*A], reshape to [H, A]
                    edit_action = edit_action.reshape(self.action_horizon, self.action_dim)
                    return edit_action  # [H, A]
                
                edit_samples = jax.vmap(edit_single_action)(edit_rngs[:self.n_edit_samples], anchor_actions)  # [n, H, A]
                edit_samples = edit_samples * self.edit_action_scale + anchor_actions
                action_samples = jnp.concatenate([action_samples, edit_samples], axis=0)  # [N+n, H, A]
            
            # Evaluate Q-values and select best action
            rng, critic_rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                critic_rng, self.target_critic.params, self.num_min_qs, self.num_qs
            )
            
            def eval_q(obs, acts):
                K, H, A = acts.shape
                # acts is already in shape (K, H, A), so no need to reshape
                obs_rep = jax.tree_util.tree_map(lambda x: x.repeat(K, axis=0), obs)
                q_flat = compute_q(self.target_critic.apply_fn, target_params, obs_rep, acts)
                return q_flat  # [K]
            
            q_values = eval_q(observations, action_samples)  # [N+n]
            best_idx = jnp.argmax(q_values)
            actions = action_samples[best_idx]  # [H, A]
        else:
            # Single sample case
            rng, actor_rng = jax.random.split(self.rng)
            actions = self.actor.sample_actions(actor_rng, observations, num_steps=self.T)

        rng, _ = jax.random.split(rng)
        return actions, self.replace(rng=rng) # [H, A]


    def sample_batch_actions(self, observations: _model.Observation) -> _model.Actions:
        observations = jax.device_put(observations)

        rngs = jax.random.split(self.rng, self.N + 1)
        rng, sample_rngs = rngs[0], rngs[1:]

        action_samples = jax.vmap(
            lambda rng: self.actor.sample_actions(rng, observations, num_steps=self.T)
        )(sample_rngs) # [N, B, H, A]
        action_samples = jnp.swapaxes(action_samples, 0, 1) # [B, N, H, A]

        if self.n_edit_samples > 0:
            edit_rngs = jax.random.split(rng, self.n_edit_samples + 1)
            rng, edit_rngs = edit_rngs[0], edit_rngs[1:]
            anchor_actions = action_samples[:, :self.n_edit_samples] # [B, n, H, A]

            def edit_single_action(rng, i):
                anchor_i = anchor_actions[:, i, :, :]  # [B, H, A] - keep batch dimension
                edit_i = _sample_actions(rng, self.edit_actor.apply_fn, self.edit_actor.params, observations, anchor_i)
                # edit_i is [B, H*A], reshape to [B, H, A]
                edit_i = edit_i.reshape(-1, self.action_horizon, self.action_dim)
                return edit_i # [B, H, A]
            
            edit_samples = jax.vmap(edit_single_action)(edit_rngs, jnp.arange(self.n_edit_samples)) # [n, B, H, A]

            edit_samples = jnp.swapaxes(edit_samples, 0, 1) # [B, n, H, A]
            edit_samples = edit_samples * self.edit_action_scale + anchor_actions
            action_samples = jnp.concatenate([action_samples, edit_samples], axis=1) # [B, N+n, H, A]

        if self.N > 1:
            rng, critic_rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                critic_rng, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            def eval_q(critic_params, obs, acts):
                B, K, H, A = acts.shape
                acts_flat = acts.reshape(B*K, H, A)
                obs_rep = jax.tree_util.tree_map(lambda x: x.repeat(K, axis=0), obs)
                q_flat = compute_q(self.target_critic.apply_fn, critic_params, obs_rep, acts_flat)
                return q_flat.reshape(B, K)

            q_values = eval_q(target_params, observations, action_samples) # [B, K]
            best_idx = jnp.argmax(q_values, axis=1) # [B]

            batch_idx = jnp.arange(action_samples.shape[0])
            best_actions = action_samples[batch_idx, best_idx, :, :]
        else:
            best_actions = action_samples[:, 0, :, :]
        
        rng, _ = jax.random.split(rng)
        return best_actions, rng # [B, H, A]   

    # 실제 환경에서 rollout할 때에 사용 (observation 한개 B=1)
    def sample_actions(self, observations: _model.Observation) -> _model.Actions:
        if self.N > 1:
            # Multiple samples case
            rngs = jax.random.split(self.rng, 1 + self.N + self.n_edit_samples)  # for critic and edit samples
            rng, sample_rngs, edit_rngs = rngs[0], rngs[1:self.N+1], rngs[self.N+1:]
            
            # Generate N action samples
            action_samples = jax.vmap(
                lambda rng: self.actor.sample_actions(rng, observations, num_steps=self.T)
            )(sample_rngs).squeeze(1)  # [N, 1, H, A] ->[N, H, A]

            # print(action_samples.shape)
            
            if self.n_edit_samples > 0:
                anchor_actions = action_samples[:self.n_edit_samples]  # [n, H, A]
                # print(anchor_actions.shape)
                
                def edit_single_action(rng, anchor_action):
                    # Add batch dimension to anchor_action: [H, A] -> [1, H, A]
                    anchor_action_batched = anchor_action[None, :, :]
                    edit_action = _sample_actions(rng, self.edit_actor.apply_fn, self.edit_actor.params, observations, anchor_action_batched)
                    # edit_action is [1, H*A], reshape to [H, A]
                    edit_action = edit_action.reshape(self.action_horizon, self.action_dim)
                    return edit_action  # [H, A]
                
                edit_samples = jax.vmap(edit_single_action)(edit_rngs[:self.n_edit_samples], anchor_actions)  # [n, H, A]
                edit_samples = edit_samples * self.edit_action_scale + anchor_actions
                action_samples = jnp.concatenate([action_samples, edit_samples], axis=0)  # [N+n, H, A]
            
            # Evaluate Q-values and select best action
            rng, critic_rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                critic_rng, self.target_critic.params, self.num_min_qs, self.num_qs
            )
            
            def eval_q(obs, acts):
                K, H, A = acts.shape
                # acts is already in shape (K, H, A), so no need to reshape
                obs_rep = jax.tree_util.tree_map(lambda x: x.repeat(K, axis=0), obs)
                q_flat = compute_q(self.target_critic.apply_fn, target_params, obs_rep, acts)
                return q_flat  # [K]
            
            q_values = eval_q(observations, action_samples)  # [N+n]
            best_idx = jnp.argmax(q_values)
            actions = action_samples[best_idx]  # [H, A]
        else:
            # Single sample case
            rng, actor_rng = jax.random.split(self.rng)
            actions = self.actor.sample_actions(actor_rng, observations, num_steps=self.T)

        rng, _ = jax.random.split(rng)
        return actions, self.replace(rng=rng) # [H, A]

    # def update_actor(self, batch: tuple[_model.Observation, _model.Actions]) -> Tuple[Agent, Dict[str, float]]:
    #     rng, actor_rng = jax.random.split(self.rng)
    #     model = self.actor.rebuild()
    #     model.train()

    #     def actor_loss_fn(model, rng, observation, actions):
    #         # actions: [batch_size, action_horizon, action_dim]
    #         # Pi0 model expects this exact shape, so no padding needed
    #         chunked_loss = model.compute_loss(rng, observation, actions, train=True)
    #         return jnp.mean(chunked_loss)

    #     train_rng = jax.random.fold_in(actor_rng, self.actor.step)
    #     observation, actions = batch

    #     diff_state = nnx.DiffState(0, self.actor.trainable_filter)
    #     loss, grads = nnx.value_and_grad(actor_loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    #     params = self.actor.params.filter(self.actor.trainable_filter)
    #     updates, new_opt_state = self.actor.tx.update(grads, self.actor.opt_state, params)
    #     new_params = optax.apply_updates(params, updates)

    #     nnx.update(model, new_params)
    #     new_params = nnx.state(model)

    #     new_state = self.actor.replace(
    #         step=self.actor.step + 1,
    #         params=new_params,
    #         opt_state=new_opt_state,
    #     )

    #     info = {
    #         "actor_loss": loss,
    #         "grad_norm": optax.global_norm(grads),
    #     }

    #     self.replace(actor=new_state, rng=rng)

    #     return info


    def update_actor(self, batch: tuple[_model.Observation, _model.Actions]) -> Tuple[Agent, Dict[str, float]]:
        rng, actor_rng = jax.random.split(self.rng)
        model = self.actor.rebuild()
        model.train()

        def actor_loss_fn(model, rng, observation, actions):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        train_rng = jax.random.fold_in(actor_rng, self.actor.step)
        observation, actions = batch

        diff_state = nnx.DiffState(0, self.actor.trainable_filter)
        loss, grads = nnx.value_and_grad(actor_loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

        params = self.actor.params.filter(self.actor.trainable_filter)
        updates, new_opt_state = self.actor.tx.update(grads, self.actor.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_actor = self.actor.replace(step=self.actor.step + 1, params=new_params, opt_state=new_opt_state)

        new_agent = self.replace(actor=new_actor, rng=rng)
        
        return new_agent, {"actor_loss": loss, "grad_norm": optax.global_norm(grads)}



    def update_edit_actor(self, batch: tuple[_model.Observation, _model.Actions]) -> Dict[str, float]:
        rng, edit_actor_rng = jax.random.split(self.rng)
        dropout_key, rng = jax.random.split(rng)
        
        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            obs, act = batch
            dist = self.edit_actor.apply_fn({"params": actor_params}, obs, act, training=True, rngs={"dropout": dropout_key})
            edit_actions = dist.sample(seed=edit_actor_rng)

            log_probs = dist.log_prob(edit_actions)
            edit_actions = edit_actions * self.edit_action_scale
            log_probs -= edit_actions.shape[-1] * jnp.log(self.edit_action_scale)

            # Reshape edit_actions from (batch_size, action_horizon*action_dim) to (batch_size, action_horizon, action_dim)
            edit_actions = edit_actions.reshape(-1, self.action_horizon, self.action_dim)
            actions = act + edit_actions

            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                obs,
                actions,
                True, # training=True
                rngs={"dropout": dropout_key},
            )
            q = qs.mean(axis=0)
            edit_actor_loss = (
                self.entropy_scale * log_probs * jax.lax.stop_gradient(self.temp.apply_fn({"params": self.temp.params})) - jax.lax.stop_gradient(q)
            ).mean()
            return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        self.replace(edit_actor=edit_actor, rng=rng)
        return actor_info


    def update_edit_actor(self, batch: tuple[_model.Observation, _model.Actions]) -> Tuple[Agent, Dict[str, float]]:
        rng, edit_actor_rng = jax.random.split(self.rng)
        obs, act = batch
        new_p, new_opt, info = self._edit_step(
            self.edit_actor.params, self.edit_actor.opt_state, edit_actor_rng,
            obs=obs, act=act,
            edit_scale=self.edit_action_scale, entropy_scale=self.entropy_scale,
            critic_params=self.critic.params, temp_params=self.temp.params,
        )
        new_edit_actor = self.edit_actor.replace(step=self.edit_actor.step + 1,params=new_p, opt_state=new_opt)
        new_agent = self.replace(edit_actor=new_edit_actor, rng=rng)
        return new_agent, info


    def update_temperature(self, entropy: float) -> Dict[str, float]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        self.replace(temp=temp)

        return temp_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        rng, temp_rng = jax.random.split(self.rng)
        new_p, new_opt, info = self._temp_step(
            self.temp.params, self.temp.opt_state, entropy=entropy, target_entropy=self.target_entropy
        )
        new_temp = self.temp.replace(step=self.temp.step + 1, params=new_p, opt_state=new_opt)
        new_agent = self.replace(temp=new_temp, rng=rng)
        return new_agent, info



    # Chunk RL implemented - using full action horizon for Q-learning
    def update_critic(self, observations, actions, rewards, next_observations, masks) -> Dict[str, float]:
        # actions: [batch_size, action_horizon, action_dim]
        # rewards: [batch_size, action_horizon] - reward sequence for chunk RL
        
        rng, critic_rng = jax.random.split(self.rng)
        
        # Sample next actions for target Q calculation
        next_actions, new_agent = self.sample_batch_actions(next_observations)
        # next_actions: [batch_size, action_horizon, action_dim]
        
        rng, target_key = jax.random.split(rng)
        target_params = subsample_ensemble(
            target_key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        rng, dropout_key = jax.random.split(rng)
        
        # Calculate n-step return using chunk RL approach
        H = self.action_horizon
        gamma_vec = jnp.power(self.discount, jnp.arange(H))[None, :]  # (1, H)
        
        # Calculate n-step return: G_n = Σ(γ^i * r_{t+i})
        G_n = jnp.sum(gamma_vec * rewards, axis=1)  # [batch_size]
        
        # Add bootstrap value if not terminal
        next_qs = new_agent.target_critic.apply_fn(
            {"params": target_params},
            next_observations,
            next_actions,
            True,
            rngs={"dropout": dropout_key},
        )  # training=True
        next_q = next_qs.min(axis=0)
        
        # Bootstrap with γ^H * Q(s_{t+H}, a_{t+H}) if not terminal
        mask = masks  # [batch_size] - continuation mask
        target_q = G_n + (self.discount ** H) * mask * next_q
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params) -> Dict[str, float]:
            qs = new_agent.critic.apply_fn(
                {"params": critic_params},
                observations,
                actions,
                True,
                rngs={"dropout": dropout_key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(new_agent.critic.params)
        critic = new_agent.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, new_agent.target_critic.params, self.tau
        )
        target_critic = new_agent.target_critic.replace(params=target_critic_params)

        self.replace(critic=critic, target_critic=target_critic, rng=rng)

        return info


    def update_critic(self, observations, actions, rewards, next_observations, masks) -> Tuple[Agent, Dict[str, float]]:
        # JIT 밖에서 next-actions 생성 (self.* 사용 가능)
        rng, critic_rng = jax.random.split(self.rng)
        next_actions, rng = self.sample_batch_actions(next_observations)  # [B,H,A]

        new_cparams, new_copt, new_tparams, info = self._critic_step(
            self.critic.params, self.critic.opt_state, self.target_critic.params, critic_rng,
            observations=observations, actions=actions, rewards=rewards,
            next_observations=next_observations, masks=masks, next_actions=next_actions,
            discount=self.discount, tau=self.tau,
        )
        new_critic = self.critic.replace(step=self.critic.step + 1, params=new_cparams, opt_state=new_copt)
        new_target_critic = self.target_critic.replace(step=self.target_critic.step + 1, params=new_tparams)
        
        
        # Update rng for next iteration
        rng, _ = jax.random.split(rng)
        new_agent = self.replace(critic=new_critic, target_critic=new_target_critic, rng=rng)
        
        return new_agent, info



    # @jax.jit
    # def update(self, batch: dict):
    #     observations, actions, rewards, next_observations, masks = batch["observations"], batch["actions"], batch["rewards"], batch["next_observations"], batch["masks"]
    #     critic_info = self.update_critic(observations, actions, rewards, next_observations, masks)    
    #     actor_info = self.update_actor((observations, actions))
        
    #     if self.n_edit_samples > 0:
    #         edit_actor_info = self.update_edit_actor((observations, actions))
    #         actor_info.update(edit_actor_info)
    #         temp_info = self.update_temperature(edit_actor_info["entropy"])
    #         actor_info.update(temp_info)
    #     return {**actor_info, **critic_info}

    # 이게 수정된건데 메모리 터짐
    # def update(self, batch: dict):
    #     obs = batch["observations"]
    #     acts = batch["actions"]
    #     rews = batch["rewards"]
    #     next_obs = batch["next_observations"]
    #     masks = batch["masks"]

    #     new_agent, critic_info = self.update_critic(obs, acts, rews, next_obs, masks)
    #     new_agent, actor_info = new_agent.update_actor((obs, acts))

    #     if self.n_edit_samples > 0:
    #         new_agent, edit_info = new_agent.update_edit_actor((obs, acts))
    #         actor_info.update(edit_info)
    #         new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
    #         actor_info.update(temp_info)

    #     return new_agent, {**actor_info, **critic_info}

    def update(self, batch: dict):
        return self._update_step(self, batch)

    @staticmethod
    # @jax.jit # 0번 인자(agent)의 버퍼를 기부
    def _update_step(agent, batch):
        # 기존 update의 본문을 여기로 이동 (가능하면 순수 함수 형태)
        obs = batch["observations"]
        acts = batch["actions"]
        rews = batch["rewards"]
        next_obs = batch["next_observations"]
        masks = batch["masks"]

        agent, critic_info = agent.update_critic(obs, acts, rews, next_obs, masks)
        agent, actor_info  = agent.update_actor((obs, acts))

        if agent.n_edit_samples > 0:
            agent, edit_info = agent.update_edit_actor((obs, acts))
            actor_info.update(edit_info)
            agent, temp_info = agent.update_temperature(edit_info["entropy"])
            actor_info.update(temp_info)

        return agent, {**actor_info, **critic_info}


    # # 안씀
    # @partial(jax.jit, static_argnames="utd_ratio")
    # def update_past(self, batch: dict, utd_ratio: int):
    #     new_agent = self
    #     for i in range(utd_ratio):
    #         def slice(x):
    #             assert x.shape[0] % utd_ratio == 0
    #             batch_size = x.shape[0] // utd_ratio
    #             return x[batch_size * i : batch_size * (i + 1)]

    #         mini_batch = jax.tree_util.tree_map(slice, batch)
    #         new_agent, critic_info = new_agent.update_critic(mini_batch)

    #     new_agent, actor_info = new_agent.update_actor(mini_batch)

    #     if self.n_edit_samples > 0:
    #         new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
    #         new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

    #         actor_info.update(temp_info)

    #     return new_agent, {**actor_info, **critic_info}