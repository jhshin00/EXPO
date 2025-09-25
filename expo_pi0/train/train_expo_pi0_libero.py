#!/usr/bin/env python
import os
import tempfile
import tensorflow as tf
import jax
from jax.experimental.compilation_cache import compilation_cache
import numpy as np
import gym
import gym.spaces



# Enable JAX compilation cache for faster repeated compilations
compilation_cache.set_cache_dir(os.path.expanduser("~/.cache/jax_compilation_cache"))

from expo_pi0.utils.wandb_logger import WandBLogger, create_exp_name

from expo_pi0.agents.expo_pi0_learner import EXPOPi0Learner
from expo_pi0.train.expo_buffer import TrajReplayBuffer
from expo_pi0.train.train_utils import trajwise_alternating_training_loop

# XLA optimization for better performance
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
xla_flags += ' --xla_gpu_all_reduce_combine_threshold_bytes=134217728'
xla_flags += ' --xla_gpu_all_gather_combine_threshold_bytes=134217728'
os.environ['XLA_FLAGS'] = xla_flags

# JAX configuration for better performance
jax.config.update('jax_compilation_cache_dir', os.path.expanduser("~/.cache/jax_compilation_cache"))
jax.config.update('jax_enable_x64', False)  # Use float32 for speed
jax.config.update('jax_default_matmul_precision', 'float32')  # Use float32 precision

PALIGEMMA_VOCAB_SIZE = 257_152

def create_expo_obs_space():
    """Create observation space in expo format"""
    return gym.spaces.Dict({
        'image': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'left_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'right_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        }),
        'image_mask': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
            'left_wrist_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
            'right_wrist_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
        }),
        'state': gym.spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32),
        'tokenized_prompt': gym.spaces.Box(0, PALIGEMMA_VOCAB_SIZE, (48,), dtype=np.int32),
        'tokenized_prompt_mask': gym.spaces.Box(0, 1, (48,), dtype=bool),
    })


def create_expo_action_space():
    """Create action space for expo"""
    return gym.spaces.Box(-1, 1, (32,), dtype=np.float32)


def main(config):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert config.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', config.batch_size)
    
    tf.config.set_visible_devices([], "GPU")

    if config.suffix:
        exp_name = create_exp_name(config.prefix, seed=config.seed) + f"_{config.suffix}"
    else:
        exp_name = create_exp_name(config.prefix, seed=config.seed)
    
    output_dir = os.path.join(os.environ['EXP'], exp_name)
    config.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('writing to output dir ', output_dir)

    group_name = config.prefix + '_' + config.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        config.prefix != '',
        config,
        config.wandb_project,
        exp_name,
        output_dir=wandb_output_dir,
        group_name=group_name
    )
    
    print("Creating EXPOPi0Learner agent...")
    print(f"  - action_dim: {config.action_dim}")
    print(f"  - action_horizon: {config.action_horizon}")
    print(f"  - max_token_len: {config.max_token_len}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - img_dim: {config.img_dim}")
    print(f"  - txt_dim: {config.txt_dim}")
    print(f"  - state_dim: {config.state_dim}")
    
    agent = EXPOPi0Learner.create(
        seed=config.seed,
        actor_lr=config.actor_lr,
        edit_actor_lr=config.edit_actor_lr,
        critic_lr=config.critic_lr,
        temp_lr=config.temp_lr,
        hidden_dims=config.hidden_dims,
        discount=config.discount,
        tau=config.tau,
        num_qs=config.num_qs,
        num_min_qs=config.num_min_qs,
        critic_dropout_rate=config.critic_dropout_rate,
        critic_weight_decay=config.critic_weight_decay,
        critic_layer_norm=config.critic_layer_norm,
        target_entropy=config.target_entropy,
        entropy_scale=config.entropy_scale,
        init_temperature=config.init_temperature,
        backup_entropy=config.backup_entropy,
        use_pnorm=config.use_pnorm,
        adjust_target_entropy=config.adjust_target_entropy,
        N=config.N,
        T=config.T,
        n_edit_samples=config.n_edit_samples,
        edit_action_scale=config.edit_action_scale,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        actor_drop=config.actor_drop,
        use_critic_pi0=config.use_critic_pi0,
        img_dim=config.img_dim,
        txt_dim=config.txt_dim,
        state_dim=config.state_dim,
        vocab_size=config.vocab_size,
        image_keys=config.image_keys,
        d_model=config.d_model,
        n_layers=config.n_layers,
        kernel_size=config.kernel_size,
        overwrite=config.overwrite,
        resume=config.resume,
        params_path=config.params_path,
    )
    print("EXPOPi0Learner agent created successfully!")
    
    print("Creating observation and action spaces...")
    # Create observation and action spaces
    observation_space = create_expo_obs_space()
    action_space = create_expo_action_space()
    print("Spaces created successfully!")
    
    print("Created spaces:")
    print(f"  Observation space: {observation_space}")
    print(f"  Action space: {action_space}")
    
    print("Creating replay buffer...")
    print(f"  - use_offline_data: {config.use_offline_data}")
    print(f"  - offline_dataset_subset_num: {config.offline_dataset_subset_num}")
    print(f"  - libero_data_dir: {config.libero_data_dir}")
    
    # Create replay buffer
    replay_buffer = TrajReplayBuffer(
        observation_space,
        action_space, 
        config.capacity, 
        config.use_offline_data, 
        config.libero_data_dir,
        config.offline_dataset_subset_num,
    )
    print("Replay buffer created successfully!")
    
    print("Starting training loop...")
    # Start training loop
    trajwise_alternating_training_loop(config, agent, replay_buffer, wandb_logger, config.perform_control_evals)
    