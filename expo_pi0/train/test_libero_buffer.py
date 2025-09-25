#!/usr/bin/env python3
"""
Test script to verify libero_90 dataset loading into TrajReplayBuffer.

This script:
1. Loads libero_90 dataset into TrajReplayBuffer
2. Verifies data structure and dimensions
3. Tests sampling functionality
4. Reports statistics about loaded data
"""

import numpy as np
import gym
import gym.spaces
from pathlib import Path
import sys
import os
import h5py

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from expo_pi0.train.expo_buffer import TrajReplayBuffer, load_all_libero_trajectories
from expo_pi0.train.expo_buffer import load_libero_trajectory
from expo_pi0.train.train_utils import get_libero_env
from libero.libero import benchmark


def create_mock_spaces():
    """Create mock observation and action spaces for testing."""
    # Create observation space that matches the flat trajectory structure
    obs_space = gym.spaces.Dict({
        'base_img': gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        'wrist_img': gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        'base_img_mask': gym.spaces.Box(low=np.array(False), high=np.array(True), shape=(), dtype=bool),
        'wrist_img_mask': gym.spaces.Box(low=np.array(False), high=np.array(True), shape=(), dtype=bool),
        'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        'tokenized_prompt': gym.spaces.Box(low=0, high=10000, shape=(48,), dtype=np.int32),
        'tokenized_prompt_mask': gym.spaces.Box(low=np.array([False]*48), high=np.array([True]*48), shape=(48,), dtype=bool),
    })
    
    # Create action space (7D for robot actions)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    
    return obs_space, action_space


def analyze_trajectory_structure(traj, traj_id=0):
    """Analyze and print trajectory structure."""
    print(f"\n=== Trajectory {traj_id} Analysis ===")
    print(f"Episode length: {traj['episode_length']}")
    print(f"Is success: {traj['is_success']}")
    print(f"Episode return: {traj['episode_return']}")
    print(f"Environment steps: {traj['env_steps']}")
    
    print(f"\nData shapes:")
    for key, value in traj.items():
        if key not in ['is_success', 'episode_return', 'episode_length', 'env_steps']:
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {type(value)} - {value}")
    
    # Check data consistency
    expected_length = traj['episode_length']
    for key, value in traj.items():
        if key not in ['is_success', 'episode_return', 'episode_length', 'env_steps'] and isinstance(value, np.ndarray):
            if len(value) != expected_length:
                print(f"  WARNING: {key} length {len(value)} != expected {expected_length}")


def load_sample_libero_trajectories(dataset_dir: str, max_trajectories: int = 10, max_token_len: int = 48) -> list:
    """
    Load a sample of trajectories from libero_90 dataset for testing.
    
    Args:
        dataset_dir: Path to the libero_90 dataset directory
        max_trajectories: Maximum number of trajectories to load
        max_token_len: Maximum token length for prompt tokenization
    
    Returns:
        List of trajectory dictionaries, each in the format expected by insert_traj
    """
    
    trajectories = []
    
    # Get all HDF5 files in the directory
    hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    hdf5_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(hdf5_files)} HDF5 files. Loading sample of {max_trajectories} trajectories...")
    
    for hdf5_file in hdf5_files:
        if len(trajectories) >= max_trajectories:
            break
            
        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        
        # Extract task description from filename
        # Format: SCENE_TASK_NAME_demo.hdf5
        task_name = hdf5_file.replace('_demo.hdf5', '')
        task_description = task_name.replace('_', ' ').title()
        
        # Load only first few demonstrations from this file
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_indices = [int(k.split('_')[1]) for k in demo_keys]
            demo_indices.sort()
            
            # Load only first 2 demos from each file
            for demo_idx in demo_indices[:2]:
                if len(trajectories) >= max_trajectories:
                    break
                    
                try:
                    traj = load_libero_trajectory(hdf5_path, demo_idx, task_description)
                    trajectories.append(traj)
                    print(f"Loaded trajectory {len(trajectories)}: {hdf5_file} demo_{demo_idx}")
                except Exception as e:
                    print(f"Warning: Failed to load {hdf5_file} demo_{demo_idx}: {e}")
                    continue
    
    print(f"Loaded {len(trajectories)} trajectories from {len(hdf5_files)} files")
    return trajectories


def test_buffer_sampling(buffer, num_samples=5):
    """Test buffer sampling functionality."""
    print(f"\n=== Testing Buffer Sampling ===")
    
    # Test trajectory sampling
    print(f"Testing get_random_trajs...")
    trajs = buffer.get_random_trajs(num_trajs=min(3, buffer.size))
    if trajs:
        print(f"  Sampled {len(trajs)} trajectories")
        for i, traj_data in enumerate(trajs):
            print(f"  Trajectory {i}: {traj_data['observations']['episode_length']} steps")
    
    # Test step sampling
    print(f"Testing sample...")
    batch = buffer.sample(batch_size=min(5, buffer.size))
    if batch:
        print(f"  Sampled batch size: {len(batch['actions'])}")
        print(f"  Observation type: {type(batch['observations'])}")
        print(f"  Action shape: {batch['actions'].shape}")
        print(f"  Reward shape: {batch['rewards'].shape}")
        
        # Check observation structure in detail
        obs = batch['observations']
        print(f"  Observation details:")
        
        # Check images
        if hasattr(obs, 'images'):
            print(f"    Images keys: {list(obs.images.keys())}")
            for k, v in obs.images.items():
                print(f"      {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")
        
        # Check image masks
        if hasattr(obs, 'image_masks'):
            print(f"    Image masks keys: {list(obs.image_masks.keys())}")
            for k, v in obs.image_masks.items():
                print(f"      {k}: shape={v.shape}, dtype={v.dtype}, values={np.unique(v)}")
        
        # Check state
        if hasattr(obs, 'state'):
            print(f"    State: shape={obs.state.shape}, dtype={obs.state.dtype}")
            print(f"      range=[{obs.state.min():.3f}, {obs.state.max():.3f}]")
            print(f"      mean={obs.state.mean(axis=0)}")
        
        # Check tokenized prompt
        if hasattr(obs, 'tokenized_prompt'):
            print(f"    Tokenized prompt: shape={obs.tokenized_prompt.shape}, dtype={obs.tokenized_prompt.dtype}")
            print(f"      range=[{obs.tokenized_prompt.min()}, {obs.tokenized_prompt.max()}]")
            print(f"      sample tokens: {obs.tokenized_prompt[0, :10]}")  # First 10 tokens of first sample
        
        # Check tokenized prompt mask
        if hasattr(obs, 'tokenized_prompt_mask'):
            print(f"    Tokenized prompt mask: shape={obs.tokenized_prompt_mask.shape}, dtype={obs.tokenized_prompt_mask.dtype}")
            print(f"      mask values: {np.unique(obs.tokenized_prompt_mask)}")
            print(f"      sample mask: {obs.tokenized_prompt_mask[0, :10]}")  # First 10 masks of first sample


def main():
    """Main test function."""
    print("=== Libero 90 Dataset Buffer Test (Sample) ===")
    
    # Dataset path
    libero_data_dir = "/ssd2/EXPO/datasets/libero_90"
    
    if not os.path.exists(libero_data_dir):
        print(f"ERROR: Dataset directory not found: {libero_data_dir}")
        print("Please check the path and ensure libero_90 dataset is downloaded.")
        return
    
    # Create mock spaces
    obs_space, action_space = create_mock_spaces()
    print(f"Created observation space: {obs_space}")
    print(f"Created action space: {action_space}")
    
    # Load sample trajectories
    print(f"\n=== Loading Sample Libero Data ===")
    sample_trajectories = load_sample_libero_trajectories(libero_data_dir, max_trajectories=5)
    
    # Initialize buffer without offline data
    print(f"\n=== Initializing Buffer ===")
    buffer = TrajReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        capacity=10000,  # 10k steps for testing
        use_offline_data=False,  # Don't auto-load
        libero_data_dir=libero_data_dir
    )
    
    # Manually insert sample trajectories
    print(f"Inserting {len(sample_trajectories)} sample trajectories...")
    for i, traj in enumerate(sample_trajectories):
        buffer.insert_traj(traj)
        print(f"Inserted trajectory {i+1}/{len(sample_trajectories)}")
    
    print(f"Buffer initialized successfully!")
    print(f"Number of trajectories loaded: {buffer.size}")
    print(f"Total steps loaded: {buffer.total_steps}")
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Buffer utilization: {buffer.total_steps / buffer.capacity * 100:.1f}%")
    
    # Analyze first few trajectories
    print(f"\n=== Analyzing Loaded Trajectories ===")
    traj_ids = list(buffer.trajectories.keys())[:5]  # First 5 trajectories
    
    for i, traj_id in enumerate(traj_ids):
        traj = buffer.trajectories[traj_id]
        analyze_trajectory_structure(traj, traj_id)
    
    # Test sampling
    test_buffer_sampling(buffer)
    
    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    episode_lengths = [buffer.trajectories[tid]['episode_length'] for tid in buffer.trajectories.keys()]
    success_rates = [buffer.trajectories[tid]['is_success'] for tid in buffer.trajectories.keys()]
    episode_returns = [buffer.trajectories[tid]['episode_return'] for tid in buffer.trajectories.keys()]
    
    print(f"Total trajectories: {len(episode_lengths)}")
    print(f"Total steps: {sum(episode_lengths)}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min episode length: {min(episode_lengths)}")
    print(f"Max episode length: {max(episode_lengths)}")
    print(f"Success rate: {np.mean(success_rates):.2%}")
    print(f"Average episode return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    
    # Check data integrity
    print(f"\n=== Data Integrity Check ===")
    all_valid = True
    for traj_id, traj in buffer.trajectories.items():
        expected_length = traj['episode_length']
        for key, value in traj.items():
            if key not in ['is_success', 'episode_return', 'episode_length', 'env_steps'] and isinstance(value, np.ndarray):
                if len(value) != expected_length:
                    print(f"ERROR: Trajectory {traj_id}, {key} length mismatch: {len(value)} != {expected_length}")
                    all_valid = False
    
    if all_valid:
        print("✓ All trajectory data is consistent!")
    else:
        print("✗ Some trajectory data has inconsistencies!")
    
    print(f"\n=== Test Complete ===")
    print(f"Buffer is ready for training with {buffer.size} trajectories and {buffer.total_steps} total steps.")


if __name__ == "__main__":
    main()
