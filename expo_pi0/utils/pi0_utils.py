from collections.abc import Callable
from typing import Any
import dataclasses

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.training.optimizer as _optimizer

import openpi.training.weight_loaders as _weight_loaders
import flax.traverse_util as traverse_util

from typing_extensions import TypeAlias
import chex
ArrayTree: TypeAlias = chex.ArrayTree


@at.typecheck
@struct.dataclass
class TrainStatePi0:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State #전체 params (trainable + frozen)
    model_def: nnx.GraphDef[_model.BaseModel]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    trainable_filter: nnx.filterlib.Filter = struct.field(pytree_node=False)


    def rebuild(self):
        return nnx.merge(self.model_def, self.params)
    
    def sample_actions(self, rng: at.KeyArrayLike, observations: _model.Observation, *, num_steps: int | at.Int[at.Array, ""] = 10) -> _model.Actions:
        sample_rng, rng = jax.random.split(rng)
        model = self.rebuild()
        return model.sample_actions(sample_rng, observations, num_steps=num_steps)
    
    @classmethod
    def create(cls, *, pi0_config, rng, overwrite = True, resume = False, params_path = None): #config : Pi0Config
        # optimizer = dataclasses.dataclass(default_factory=_optimizer.AdamW)
        optimizer = _optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        )
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps=1_00,
            peak_lr=1.0e-5,
            decay_steps=40_000,
            decay_lr=1.0e-6,
        )
        tx = _optimizer.create_optimizer(optimizer, lr_schedule, weight_decay_mask=None)
        def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> cls:
            rng, model_rng = jax.random.split(rng)
            # initialize the model (and its parameters).
            model = pi0_config.create(model_rng)

            # Merge the partial params into the model.
            if partial_params is not None:
                graphdef, state = nnx.split(model)
                # This will produce an error if the partial params are not a subset of the state.
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            # Convert frozen params to bfloat16.
            params = state_map(params, pi0_config.get_freeze_filter(), lambda p: p.replace(p.value.astype(jnp.bfloat16)))

            trainable_filter = nnx.All(nnx.Param, nnx.Not(pi0_config.get_freeze_filter()))

            return cls(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=tx,
                opt_state=tx.init(params.filter(trainable_filter)),
                trainable_filter=trainable_filter,
            )

        train_state_shape = jax.eval_shape(init, rng)

        if resume:
            return train_state_shape
        
        weight_loader = _weight_loaders.CheckpointWeightLoader(params_path)
        partial_params = _load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())

        train_state = jax.jit(
            init,
            donate_argnums=(1,),
        )(rng, partial_params)
        
        return train_state

# Checkpoint utilities
def save_checkpoint(state: TrainStatePi0, checkpoint_path: str):
    """Save checkpoint to disk."""
    import pickle
    checkpoint_data = {
        'step': state.step,
        'params': state.params.to_pure_dict(),
        'opt_state': state.opt_state,
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_path: str, tx: optax.GradientTransformation, pi0_config) -> TrainStatePi0:
    """Load checkpoint from disk."""
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Rebuild model from checkpoint
    trainable_filter = nnx.All(nnx.Param, nnx.Not(pi0_config.get_freeze_filter()))
    model_def = pi0_config.create(jax.random.PRNGKey(0))
    model = nnx.merge(model_def, checkpoint_data['params'])
    params = nnx.state(model)
    
    return TrainStatePi0(
        step=checkpoint_data['step'],
        params=params,
        model_def=model_def,
        tx=tx,
        opt_state=checkpoint_data['opt_state'],
        trainable_filter=trainable_filter,
    )


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """Apply a function to the leaves of the state that match the filter."""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )