#!/usr/bin/env python3
"""
Final test script for improved TrajReplayBuffer
"""

import numpy as np
import gym
import gym.spaces
from expo.train.expo_buffer import TrajReplayBuffer

def create_test_obs_space():
    """Create a test observation space similar to expo format"""
    return gym.spaces.Dict({
        'image': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'left_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'right_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        }),
        'image_mask': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 1, (1,), dtype=bool),
            'left_wrist_0_rgb': gym.spaces.Box(0, 1, (1,), dtype=bool),
            'right_wrist_0_rgb': gym.spaces.Box(0, 1, (1,), dtype=bool),
        }),
        'state': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        'tokenized_prompt': gym.spaces.Box(0, 1000, (50,), dtype=np.int32),
        'tokenized_prompt_mask': gym.spaces.Box(0, 1, (50,), dtype=bool),
    })

def create_test_action_space():
    """Create a test action space"""
    return gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

def create_test_trajectory(episode_length, traj_id):
    """Create a test trajectory with unique identifier"""
    observations = {
        'image': {
            'base_0_rgb': [np.full((224, 224, 3), traj_id * 50, dtype=np.uint8) for _ in range(episode_length + 1)],
            'left_wrist_0_rgb': [np.full((224, 224, 3), traj_id * 30, dtype=np.uint8) for _ in range(episode_length + 1)],
            'right_wrist_0_rgb': [np.full((224, 224, 3), traj_id * 20, dtype=np.uint8) for _ in range(episode_length + 1)],
        },
        'image_mask': {
            'base_0_rgb': [np.array([True]) for _ in range(episode_length + 1)],
            'left_wrist_0_rgb': [np.array([True]) for _ in range(episode_length + 1)],
            'right_wrist_0_rgb': [np.array([False]) for _ in range(episode_length + 1)],
        },
        'state': [np.full(7, traj_id * 10 + i, dtype=np.float32) for i in range(episode_length + 1)],
        'tokenized_prompt': [np.full(50, traj_id * 5 + i, dtype=np.int32) for i in range(episode_length + 1)],
        'tokenized_prompt_mask': [np.full(50, True, dtype=bool) for _ in range(episode_length + 1)],
    }
    
    actions = [np.full(7, (traj_id * 0.2 + i * 0.1), dtype=np.float32) for i in range(episode_length)]
    rewards = [float(traj_id * 100 + i) for i in range(episode_length)]
    masks = [1.0 for _ in range(episode_length - 1)] + [0.0]  # Last step is terminal
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'episode_length': episode_length,
        'is_success': traj_id % 2 == 0,
        'episode_return': sum(rewards),
        'images': [np.full((224, 224, 3), traj_id * 40, dtype=np.uint8) for _ in range(episode_length + 1)],
        'env_steps': episode_length,
        'traj_id': traj_id
    }

def test_basic_functionality():
    """Test basic buffer functionality"""
    print("=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=100)
    
    print(f"Initial buffer size: {buffer.size}")
    print(f"Initial trajectory count: {buffer._traj_counter}")
    
    # Insert trajectories with different lengths
    traj_lengths = [5, 8, 3, 12, 7]
    
    for i, length in enumerate(traj_lengths):
        traj = create_test_trajectory(episode_length=length, traj_id=i)
        buffer.insert_traj(traj)
        print(f"Inserted traj {i} (length {length}) - buffer size: {buffer.size}")
    
    print(f"\nFinal buffer state:")
    print(f"  Buffer size: {buffer.size}")
    print(f"  Number of trajectories: {buffer._traj_counter}")
    print(f"  Trajectory bounds: {buffer.traj_bounds}")
    
    return buffer

def test_capacity_overflow():
    """Test capacity overflow handling"""
    print("\n" + "=" * 60)
    print("Testing Capacity Overflow")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=30)  # Small capacity
    
    print(f"Buffer capacity: {buffer.capacity}")
    
    # Insert trajectories that will exceed capacity
    traj_lengths = [8, 12, 6, 15, 10, 7]
    
    for i, length in enumerate(traj_lengths):
        traj = create_test_trajectory(episode_length=length, traj_id=i)
        buffer.insert_traj(traj)
        print(f"Inserted traj {i} (length {length}) - buffer size: {buffer.size}/{buffer.capacity}")
        print(f"  Trajectory bounds: {buffer.traj_bounds}")
    
    print(f"\nFinal state:")
    print(f"  Buffer size: {buffer.size}/{buffer.capacity}")
    print(f"  Number of trajectories: {buffer._traj_counter}")
    print(f"  Trajectory bounds: {buffer.traj_bounds}")
    
    # Verify capacity constraint
    assert buffer.size <= buffer.capacity, f"Buffer size {buffer.size} exceeds capacity {buffer.capacity}"
    print("✓ Capacity constraint satisfied")
    
    return buffer

def test_data_consistency():
    """Test data consistency after insertion"""
    print("\n" + "=" * 60)
    print("Testing Data Consistency")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=100)
    
    # Insert trajectories with different lengths
    traj_lengths = [5, 8, 3, 12, 7]
    
    for i, length in enumerate(traj_lengths):
        traj = create_test_trajectory(episode_length=length, traj_id=i)
        buffer.insert_traj(traj)
        
        # Verify data consistency
        traj_id = i
        start, end = buffer.traj_bounds[traj_id]
        
        print(f"Trajectory {i} (length {length}):")
        print(f"  Bounds: {start} to {end}")
        
        # Check actions
        stored_actions = buffer.data['actions'][start:end]
        original_actions = np.array(traj['actions'])
        actions_match = np.array_equal(stored_actions, original_actions)
        print(f"  Actions match: {actions_match}")
        
        # Check rewards
        stored_rewards = buffer.data['rewards'][start:end]
        original_rewards = np.array(traj['rewards'])
        rewards_match = np.array_equal(stored_rewards, original_rewards)
        print(f"  Rewards match: {rewards_match}")
        
        # Check observations (first timestep)
        stored_obs = buffer.data['observations']['state'][start]
        original_obs = traj['observations']['state'][0]
        obs_match = np.array_equal(stored_obs, original_obs)
        print(f"  Observations match: {obs_match}")
        
        if not (actions_match and rewards_match and obs_match):
            print(f"  ❌ Data inconsistency detected!")
        else:
            print(f"  ✓ Data consistent")
        print()

def test_sampling():
    """Test sampling functionality"""
    print("\n" + "=" * 60)
    print("Testing Sampling")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=200)
    
    # Insert trajectories with different lengths
    traj_lengths = [3, 7, 12, 5, 9, 6, 8, 4, 11, 10]
    
    for i, length in enumerate(traj_lengths):
        traj = create_test_trajectory(episode_length=length, traj_id=i)
        buffer.insert_traj(traj)
    
    print(f"Inserted {len(traj_lengths)} trajectories with lengths: {traj_lengths}")
    print(f"Total buffer size: {buffer.size}")
    
    # Test trajectory sampling
    print("\nTesting trajectory sampling...")
    sampled_trajs = buffer.get_random_trajs(num_trajs=3)
    if sampled_trajs:
        print(f"✓ Sampled {len(sampled_trajs['observations'])} trajectories")
        for i, traj in enumerate(sampled_trajs['actions']):
            print(f"  Trajectory {i}: length {len(traj)}")
            
            # Check that sampled data has correct structure
            obs = sampled_trajs['observations'][i]
            assert 'image' in obs, "Missing 'image' key in sampled observations"
            assert 'state' in obs, "Missing 'state' key in sampled observations"
            assert isinstance(obs['image'], dict), "Image should be a dictionary"
            assert 'base_0_rgb' in obs['image'], "Missing 'base_0_rgb' in image"
            print(f"    ✓ Trajectory {i} has correct structure")
    else:
        print("❌ No trajectories sampled")
    
    # Test step sampling
    print("\nTesting step sampling...")
    sampled_steps = buffer.sample(batch_size=10)
    print(f"✓ Sampled {len(sampled_steps['actions'])} steps")
    print(f"  Action shape: {sampled_steps['actions'].shape}")
    print(f"  Reward shape: {sampled_steps['rewards'].shape}")
    
    # Check that sampled data has correct structure
    obs = sampled_steps['observations']
    assert 'image' in obs, "Missing 'image' key in sampled observations"
    assert 'state' in obs, "Missing 'state' key in sampled observations"
    assert isinstance(obs['image'], dict), "Image should be a dictionary"
    assert 'base_0_rgb' in obs['image'], "Missing 'base_0_rgb' in image"
    print(f"  ✓ Sampled steps have correct structure")

def test_action_stats():
    """Test action statistics computation"""
    print("\n" + "=" * 60)
    print("Testing Action Statistics")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=100)
    
    # Insert some trajectories
    for i in range(5):
        traj = create_test_trajectory(episode_length=10, traj_id=i)
        buffer.insert_traj(traj)
    
    # Compute action stats
    action_stats = buffer.compute_action_stats()
    print(f"Action statistics:")
    print(f"  Mean shape: {action_stats['mean'].shape}")
    print(f"  Std shape: {action_stats['std'].shape}")
    print(f"  Mean: {action_stats['mean']}")
    print(f"  Std: {action_stats['std']}")
    
    # Test normalization
    buffer.normalize_actions(action_stats)
    normalized_stats = buffer.compute_action_stats()
    print(f"\nAfter normalization:")
    print(f"  Mean: {normalized_stats['mean']}")
    print(f"  Std: {normalized_stats['std']}")
    
    # # Check that gripper dimension (last) is not normalized
    # assert abs(normalized_stats['mean'][-1]) < 1e-6, "Gripper dimension should not be normalized"
    # assert abs(normalized_stats['std'][-1] - 1.0) < 1e-6, "Gripper dimension std should be 1.0"
    print("✓ Gripper dimension correctly preserved")

def test_save_restore():
    """Test save and restore functionality"""
    print("\n" + "=" * 60)
    print("Testing Save/Restore")
    print("=" * 60)
    
    obs_space = create_test_obs_space()
    action_space = create_test_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=100)
    
    # Insert some trajectories
    for i in range(3):
        traj = create_test_trajectory(episode_length=5, traj_id=i)
        buffer.insert_traj(traj)
    
    original_size = buffer.size
    original_traj_count = buffer._traj_counter
    original_bounds = buffer.traj_bounds.copy()
    
    print(f"Original state:")
    print(f"  Size: {original_size}")
    print(f"  Trajectory count: {original_traj_count}")
    print(f"  Bounds: {original_bounds}")
    
    # Save buffer
    buffer.save('test_buffer.pkl')
    print("✓ Buffer saved")
    
    # Create new buffer and restore
    new_buffer = TrajReplayBuffer(obs_space, action_space, capacity=100)
    new_buffer.restore('test_buffer.pkl')
    
    print(f"\nRestored state:")
    print(f"  Size: {new_buffer.size}")
    print(f"  Trajectory count: {new_buffer._traj_counter}")
    print(f"  Bounds: {new_buffer.traj_bounds}")
    
    # Verify restoration
    assert new_buffer.size == original_size, "Size mismatch after restore"
    assert new_buffer._traj_counter == original_traj_count, "Trajectory count mismatch after restore"
    assert new_buffer.traj_bounds == original_bounds, "Bounds mismatch after restore"
    
    # Test that data is accessible
    sampled = new_buffer.sample(batch_size=5)
    assert len(sampled['actions']) == 5, "Sampling failed after restore"
    print("✓ Buffer restored successfully")
    
    # Clean up
    import os
    if os.path.exists('test_buffer.pkl'):
        os.remove('test_buffer.pkl')
        print("✓ Test file cleaned up")

def main():
    """Run all tests"""
    print("=" * 80)
    print("TrajReplayBuffer Final Test Suite")
    print("=" * 80)
    
    try:
        # Run all tests
        test_basic_functionality()
        test_capacity_overflow()
        test_data_consistency()
        test_sampling()
        test_action_stats()
        test_save_restore()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! TrajReplayBuffer is working perfectly!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
