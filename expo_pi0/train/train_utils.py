import numpy as np
import pathlib
from tqdm import tqdm
import warnings
from typing import Any, Callable
import pathlib

import jax
import jax.numpy as jnp
from jax import jit
import wandb
import logging
import imageio

# Suppress JAX and Flax deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow_probability")

from expo_pi0.utils.wandb_logger import WandBLogger, create_exp_name

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.shared import array_typing as at
from openpi.transforms import TokenizePrompt
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x

def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-6:
        return np.zeros(3)

    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den

@jit
def _quat2axisangle_jax(quat):
    """JAX-compiled quaternion to axis-angle conversion"""
    quat = jnp.clip(quat, -1.0, 1.0)
    den = jnp.sqrt(1.0 - quat[3] * quat[3])
    return jnp.where(
        jnp.abs(den) < 1e-6,
        jnp.zeros(3),
        (quat[:3] * 2.0 * jnp.arccos(quat[3])) / den
    )

def obs_to_expo_pi0_format_dict(obs, task_description, max_token_len, action_dim=7):
    base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    base_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(base_img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )
    images = {
        "base_0_rgb": base_img[None, ...],
        "left_wrist_0_rgb": wrist_img[None, ...],
        "right_wrist_0_rgb": np.zeros_like(base_img)[None, ...],
    }
    image_masks = {
        "base_0_rgb": np.array([True]),
        "left_wrist_0_rgb": np.array([True]),
        "right_wrist_0_rgb": np.array([False]),
    }

    # Convert quaternion using JAX for better performance
    quat_jax = jnp.array(obs["robot0_eef_quat"])
    axis_angle = _quat2axisangle_jax(quat_jax)
    
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            np.array(axis_angle),
            obs["robot0_gripper_qpos"],
        )
    )[None, ...].astype(np.float32)[:, :-1]
    
    # Pad state to match action_dim
    state = pad_to_dim(state, action_dim, axis=-1)
    prompt = str(task_description)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
    obs_dict = TokenizePrompt(tokenizer)({
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "prompt": prompt,
    })
    obs_dict["tokenized_prompt"] = obs_dict["tokenized_prompt"][None, ...]
    obs_dict["tokenized_prompt_mask"] = obs_dict["tokenized_prompt_mask"][None, ...]
    
    return obs_dict

def obs_to_expo_pi0_format(obs, task_description, max_token_len, action_dim=7):
    obs_dict = obs_to_expo_pi0_format_dict(obs, task_description, max_token_len, action_dim)
    
    # Convert numpy arrays to JAX arrays for type compatibility (more efficient)
    obs_dict["image"] = {k: jax.numpy.asarray(v) for k, v in obs_dict["image"].items()}
    obs_dict["image_mask"] = {k: jax.numpy.asarray(v) for k, v in obs_dict["image_mask"].items()}
    obs_dict["state"] = jax.numpy.asarray(obs_dict["state"])
    obs_dict["tokenized_prompt"] = jax.numpy.asarray(obs_dict["tokenized_prompt"])
    obs_dict["tokenized_prompt_mask"] = jax.numpy.asarray(obs_dict["tokenized_prompt_mask"])
    
    return _model.Observation.from_dict(obs_dict)

def obs_dict_to_expo_pi0_format(obs_dict):
    obs_dict["image"] = {k: jax.numpy.array(v) for k, v in obs_dict["image"].items()}
    obs_dict["image_mask"] = {k: jax.numpy.array(v) for k, v in obs_dict["image_mask"].items()}
    obs_dict["state"] = jax.numpy.array(obs_dict["state"])
    obs_dict["tokenized_prompt"] = jax.numpy.array(obs_dict["tokenized_prompt"])
    obs_dict["tokenized_prompt_mask"] = jax.numpy.array(obs_dict["tokenized_prompt_mask"])

    return _model.Observation.from_dict(obs_dict)


def convert_obs_list_to_flat_trajectory(obs_list, action_list, rewards, masks, is_success, episode_length, env_steps):
    """
    Convert list of observations to flat trajectory structure.
    
    Args:
        obs_list: List of observation dictionaries from to_dict()
        action_list: Array of actions
        rewards: Array of rewards
        masks: Array of masks
        is_success: Boolean success flag
        episode_length: Length of episode
        env_steps: Number of environment steps
    
    Returns:
        Flat trajectory dictionary
    """
    traj = {}
    
    # Process images - extract specific keys and concatenate
    image_keys = {
        'base_img': 'base_0_rgb',
        'wrist_img': 'left_wrist_0_rgb'
    }
    
    for traj_key, obs_key in image_keys.items():
        traj[traj_key] = np.concatenate([v["image"][obs_key] for v in obs_list], axis=0)
    
    # Process image masks
    mask_keys = {
        'base_img_mask': 'base_0_rgb',
        'wrist_img_mask': 'left_wrist_0_rgb'
    }
    
    for traj_key, obs_key in mask_keys.items():
        traj[traj_key] = np.concatenate([v["image_mask"][obs_key] for v in obs_list], axis=0)
    
    # Process other observation fields
    other_fields = ['state', 'tokenized_prompt', 'tokenized_prompt_mask']
    for field in other_fields:
        traj[field] = np.concatenate([v[field] for v in obs_list], axis=0)
    
    # Process actions, rewards, masks
    traj["actions"] = action_list
    traj["rewards"] = rewards
    traj["masks"] = masks
    
    # Add metadata
    traj["is_success"] = is_success
    traj["episode_return"] = np.sum(rewards)
    traj["episode_length"] = episode_length
    traj["env_steps"] = env_steps
    
    return traj


def collect_trajectory_data(env, agent, task_description, config):
    """
    Collect trajectory data from environment.
    
    Returns:
        obs_list: List of observation dictionaries
        action_list: Array of actions
        rewards: Array of rewards
        masks: Array of masks
        is_success: Boolean success flag
        episode_length: Length of episode
        env_steps: Number of environment steps
    """
    query_frequency = config.action_horizon
    max_timesteps = config.max_timesteps
    env_max_reward = config.env_max_reward

    obs = env.reset()
    
    # Pre-allocate lists for better performance
    obs_list = []
    action_list = []
    rewards = []
    masks = []

    pbar = tqdm(range(max_timesteps), desc="Collecting trajectory")
    for t in pbar:  
        expo_obs = obs_to_expo_pi0_format(obs, task_description, agent.max_token_len, config.action_dim)

        if t % query_frequency == 0:
            action = agent.sample_actions(expo_obs) # [H, A]
        
        action_idx = t % query_frequency
        curr_action = action[action_idx]
        
        # If action is 32D, take only first 7 dimensions for environment
        if len(curr_action) > 7:
            curr_action = curr_action[:7]
        
        curr_action = np.clip(curr_action, -1.0, 1.0)
        next_obs, reward, done, _ = env.step(curr_action)

        obs_dict = expo_obs.to_dict()
        obs_list.append(obs_dict)
        action_list.append(curr_action)
        rewards.append(reward)
        masks.append(1.0 if not done else 0.0)

        obs = next_obs
        if done:
            break

    # Add last observation
    expo_obs = obs_to_expo_pi0_format(obs, task_description, agent.max_token_len, config.action_dim)
    obs_dict = expo_obs.to_dict()
    obs_list.append(obs_dict)
    
    # Convert lists to arrays
    action_list = np.array(action_list)
    rewards = np.array(rewards)
    masks = np.array(masks)
    
    is_success = (reward == env_max_reward)
    episode_length = len(rewards)
    env_steps = t + 1
    
    return obs_list, action_list, rewards, masks, is_success, episode_length, env_steps


def get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def collect_trajectory(config, agent, env, task_description):
    """
    Collect a complete trajectory from the environment.
    
    Args:
        config: Configuration object
        agent: Agent for action sampling
        env: Environment
        task_description: Task description string
    
    Returns:
        Flat trajectory dictionary
    """
    # Collect trajectory data
    obs_list, action_list, rewards, masks, is_success, episode_length, env_steps = collect_trajectory_data(
        env, agent, task_description, config
    )
    
    # Convert to flat trajectory structure
    traj = convert_obs_list_to_flat_trajectory(
        obs_list, action_list, rewards, masks, is_success, episode_length, env_steps
    )
    
    return traj

def add_online_data_to_buffer(traj, online_replay_buffer):
    online_replay_buffer.insert_traj(traj)

def perform_control_eval(config, agent, env, task_description, wandb_logger, step):
    query_frequency = config.action_horizon
    max_timesteps = config.max_timesteps
    env_max_reward = config.env_max_reward
    success_rates = []
    episode_returns = []
    episode_lens = []
    
    for rollout_id in range(config.eval_episodes):
        image_list = []
        rewards = []

        obs = env.reset()
        
        for t in tqdm(range(max_timesteps + config.num_steps_wait), desc="Evaluating trajectory"):
            expo_obs = obs_to_expo_pi0_format(obs, task_description, agent.max_token_len, config.action_dim)

            if t < config.num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if (t - config.num_steps_wait) % query_frequency == 0:
                action, agent = agent.sample_actions(expo_obs) # [H, A]
                
            action_idx = (t - config.num_steps_wait) % query_frequency
            curr_action = action[action_idx]
            
            # If action is 32D, take only first 7 dimensions for environment
            if len(curr_action) > 7:
                curr_action = curr_action[:7]
            
            curr_action = np.clip(curr_action, -1.0, 1.0)
            next_obs, reward, done, _ = env.step(curr_action)
                
            rewards.append(reward)
                
            # Use the same image processing as in obs_to_expo_pi0_format_dict for consistency
            base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            base_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(base_img, 224, 224)
            )
            image_list.append(base_img)

            obs = next_obs
            if done:
                break
        
        # Convert to numpy arrays
        rewards = np.array(rewards)
        
        episode_lens.append(t + 1)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
        print(f'Rollout Done: {episode_return=}, Success: {is_success}')
        
        # Convert image list to proper video format: (T, H, W, C)
        video = np.stack(image_list, axis=0)  # Shape: (T, H, W, C)
        video_for_wandb = video.transpose(0, 3, 1, 2)  # Shape: (T, C, H, W) for wandb
        print(f'Video shape: {video.shape}')  # Debug print
        wandb_logger.log({f'eval_video/{task_description}': wandb.Video(video_for_wandb, fps=50)}, step=step)
        imageio.mimwrite(
            pathlib.Path(config.experiments_dir) / f'eval_video/{task_description}.mp4',
            video,
            fps=50,
        )
    
    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=step)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=step)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=step)


def trajwise_alternating_training_loop(
    config,
    agent,
    replay_buffer,
    wandb_logger,
    perform_control_evals=True,
):
    replay_buffer_iterator = replay_buffer.get_iterator(
        config.batch_size, 
        action_horizon=config.action_horizon, 
        action_dim=config.action_dim
    )
    batch = next(replay_buffer_iterator)
    logging.info(f"Initialized data loader:\n{array_tree_to_info(batch)}")

    step = 0
    start_step = 0
    wandb_logger.log({'num_online_samples': 0}, step=step)
    wandb_logger.log({'num_online_trajs': 0}, step=step)
    wandb_logger.log({'env_steps': 0}, step=step)

    # Pre-load task suite for efficiency
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config.libero_task_suite]()
    num_tasks = task_suite.get_num_tasks()
    
    # Reuse environment for multiple episodes to reduce overhead
    env_reuse_frequency = config.env_reuse_frequency  # Reuse environment for 10 episodes to reduce overhead
    env = None
    task_description = None
    
    # Get offline_steps from config, default to 0 if not specified
    offline_steps = getattr(config, 'offline_steps', 0)
    
    # pbar = tqdm(
    #     range(start_step, config.max_steps),
    #     initial=start_step,
    #     total=config.max_steps,
    #     dynamic_ncols=True,
    # )
    with tqdm(total=config.max_steps, initial=0) as pbar:
        while step < config.max_steps:
            # Determine if we're in offline learning phase
            is_offline_phase = step < offline_steps
            
            # Only collect online data if not in offline phase
            if not is_offline_phase:
                # Create new environment every env_reuse_frequency episodes
                if step % env_reuse_frequency == 0 or env is None:
                    task_id = np.random.randint(0, num_tasks)
                    task = task_suite.get_task(task_id)
                    env, task_description = get_libero_env(task, 256, config.seed)
                    eval_env = env

                traj = collect_trajectory(config, agent, env, task_description)
                add_online_data_to_buffer(traj, replay_buffer)
            else:
                # During offline phase, create a dummy trajectory for logging consistency
                traj = {
                    'is_success': False,
                    'episode_return': 0.0,
                    'episode_length': 0,
                    'env_steps': 0
                }
            
            if len(replay_buffer) > config.start_updates:
                for _ in range(config.num_update_steps):
                    if step == 0:
                        if perform_control_evals and not is_offline_phase:
                            print('performing evaluation for initial checkpoint')
                            perform_control_eval(config, agent, eval_env, task_description, wandb_logger, step)

                    batch = next(replay_buffer_iterator)
                    agent, update_info = agent.update(batch) #utd_ratio=config.utd_ratio)

                    pbar.update()
                    step += 1

                    if step % config.log_interval == 0:
                        # Only get scalar values to avoid memory accumulation
                        scalar_info = {}
                        for k, v in update_info.items():
                            if hasattr(v, 'ndim') and v.ndim == 0:
                                scalar_info[k] = float(v)  # Remove jax.device_get for speed
                            elif hasattr(v, 'item'):  # Handle scalar arrays
                                scalar_info[k] = float(v.item())
                        
                        for k, v in scalar_info.items():
                            wandb_logger.log({f'training/{k}': v}, step=step)
                        
                        # Log learning phase information
                        wandb_logger.log({
                            'learning_phase': 'offline' if is_offline_phase else 'online',
                            'replay_buffer_size': len(replay_buffer),
                            'is_success (exploration)': int(traj['is_success']) if not is_offline_phase else 0,
                        }, step=step)

                    if step % config.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': replay_buffer.total_steps}, step=step)
                        wandb_logger.log({'num_online_trajs': replay_buffer.size}, step=step)
                        if perform_control_evals and not is_offline_phase:
                            perform_control_eval(config, agent, eval_env, task_description, wandb_logger, step)
                    
                    if config.checkpoint_interval != -1 and step % config.checkpoint_interval == 0:
                        agent.save_checkpoint(config.outputdir, step, config.checkpoint_interval)


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")