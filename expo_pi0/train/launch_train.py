import sys
import os
import warnings

os.environ['EXP'] = os.path.expanduser('/ssd2/EXPO/logs')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from absl import app, flags
import uuid
from expo_pi0.train.train_expo_pi0_libero import main

# Fix CuDNN library path issue
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

# Suppress JAX and Flax deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")

# JAX configuration for better performance
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

FLAGS = flags.FLAGS

PALIGEMMA_VOCAB_SIZE = 257_152

# --- General ---
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_string('launch_group_id', str(uuid.uuid4()), 'Unique ID for the launch group.')
flags.DEFINE_string('libero_data_dir', '/ssd2/EXPO/datasets/libero_goal', 'Path to libero data directory.')
flags.DEFINE_string('libero_task_suite', 'libero_goal', 'Libero task suite.')

# --- WandB ---
flags.DEFINE_string('prefix', 'expo_pi0', 'Prefix for experiment name.')
flags.DEFINE_string('suffix', '', 'Suffix for experiment name.')
flags.DEFINE_string('wandb_project', 'expo_project', 'WandB project name.')
flags.DEFINE_string('wandb_entity', None, 'WandB entity (user or team).')

# --- Agent/Model Hyperparameters ---
# --- Actor Pi0 ---
flags.DEFINE_integer('action_dim', 32, 'Action dimension for Pi0.')
flags.DEFINE_integer('action_horizon', 50, 'Action horizon for Pi0.')
flags.DEFINE_integer('max_token_len', 48, 'Maximum token length for Pi0.')
flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate.') # 이거 안쓰고 기존 pi0의 optimizer lr 사용
flags.DEFINE_float('discount_factor', 0.99, 'Discount factor for n-step rewards.')
flags.DEFINE_boolean('overwrite', True, 'Overwrite existing weights.')
flags.DEFINE_boolean('resume', False, 'Resume training from checkpoint.')
flags.DEFINE_string('params_path', "gs://openpi-assets/checkpoints/pi0_base/params", 'Path to existing weights.')

# --- Edit Actor-Critic ---
flags.DEFINE_float('edit_actor_lr', 1.0e-4, 'Edit actor learning rate.')
flags.DEFINE_float('temp_lr', 3.0e-4, 'Temperature learning rate.')
flags.DEFINE_float('critic_lr', 1.0e-4, 'Critic learning rate.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden layer dimensions.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_float('tau', 0.005, 'Soft update coefficient (tau).')
flags.DEFINE_integer('num_qs', 1, 'Number of Q-functions.')
flags.DEFINE_integer('num_min_qs', 1, 'Number of Q-functions to use for min.')
flags.DEFINE_float('critic_dropout_rate', 0.1, 'Dropout rate for critic.')
flags.DEFINE_float('critic_weight_decay', None, 'Weight decay for critic.')
flags.DEFINE_boolean('critic_layer_norm', True, 'Use layer norm in critic.')
flags.DEFINE_float('target_entropy', None, 'Target entropy for SAC.') # 안씀
flags.DEFINE_float('entropy_scale', 6.25e-4, 'Scale for entropy.')
flags.DEFINE_float('init_temperature', 0.1, 'Initial temperature for SAC.')
flags.DEFINE_boolean('backup_entropy', True, 'Use entropy in backup.')
flags.DEFINE_boolean('use_pnorm', False, 'Use p-norm for critic regularization.')
flags.DEFINE_boolean('adjust_target_entropy', False, 'Adjust target entropy based on action dim.') # 안씀
flags.DEFINE_float('actor_drop', 0.1, 'Dropout rate for actor.')
flags.DEFINE_boolean('use_critic_pi0', False, 'Use pi0 feature for critic network.')

# --- Critic Network ---
flags.DEFINE_integer('img_dim', 256, 'Image dimension for critic network.')
flags.DEFINE_integer('txt_dim', 256, 'Text dimension for critic network.')
flags.DEFINE_integer('state_dim', 64, 'State dimension for critic network.')
flags.DEFINE_integer('vocab_size', PALIGEMMA_VOCAB_SIZE, 'Vocabulary size for critic network.')
flags.DEFINE_list('image_keys', ['base_0_rgb', 'left_wrist_0_rgb'], 'Image keys for critic network.')
flags.DEFINE_integer('d_model', 256, 'Dimension for critic network.')
flags.DEFINE_integer('n_layers', 2, 'Number of layers for critic network.')
flags.DEFINE_integer('kernel_size', 3, 'Kernel size for critic network.')

# --- EXPO Specific ---
flags.DEFINE_integer('N', 4, 'Number of samples for EXPO.')
flags.DEFINE_integer('T', 10, 'Time horizon for EXPO.')
flags.DEFINE_integer('n_edit_samples', 2, 'Number of edit samples.')
flags.DEFINE_float('edit_action_scale', 0.05, 'Scale for edit actions.')

# --- Training Loop ---
flags.DEFINE_integer('max_steps', 40_000, 'Maximum number of training steps.')
flags.DEFINE_integer('start_updates', 0, 'Number of samples before starting updates.')
flags.DEFINE_integer('num_update_steps', 100, 'Number of update steps per trajectory.')
flags.DEFINE_integer('capacity', 100000, 'Replay buffer capacity.')
flags.DEFINE_boolean('use_offline_data', True, 'Put offline data into the replay buffer.')
flags.DEFINE_integer('offline_dataset_subset_num', 10, 'Number of trajectories to sample from offline dataset (None for all).')
flags.DEFINE_boolean('perform_control_evals', True, 'Perform control evaluations during training.')
flags.DEFINE_integer('eval_episodes', 1, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100, 'Eval interval.')
flags.DEFINE_integer('checkpoint_interval', -1, 'Checkpoint interval.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_integer('env_reuse_frequency', 1, 'Number of episodes to reuse environment.')
flags.DEFINE_integer('offline_steps', 2000, 'Number of steps to perform offline learning.')

# --- Environment Parameters ---
flags.DEFINE_integer('max_timesteps', 400, 'Maximum timesteps per episode.')
flags.DEFINE_float('env_max_reward', 1.0, 'Maximum reward for environment success.')
flags.DEFINE_integer('num_steps_wait', 10, 'Number of steps to wait for objects to stabilize in sim.')

# --- Output Directory ---
flags.DEFINE_string('outputdir', './checkpoints', 'Output directory for checkpoints.')
flags.DEFINE_string('experiments_dir', './experiments', 'Experiments directory.')


def main_app(_):
    from ml_collections import ConfigDict
    config = ConfigDict(FLAGS.flag_values_dict())
    main(config)

if __name__ == '__main__':
    app.run(main_app)