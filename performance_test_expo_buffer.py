#!/usr/bin/env python3
"""
Performance test for TrajReplayBuffer
"""

import time
import numpy as np
import gym
import gym.spaces
from expo.train.expo_buffer import TrajReplayBuffer

def create_performance_obs_space():
    """Create observation space for performance testing"""
    return gym.spaces.Dict({
        'image': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'left_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        }),
        'image_mask': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 1, (1,), dtype=bool),
            'left_wrist_0_rgb': gym.spaces.Box(0, 1, (1,), dtype=bool),
        }),
        'state': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        'tokenized_prompt': gym.spaces.Box(0, 1000, (50,), dtype=np.int32),
        'tokenized_prompt_mask': gym.spaces.Box(0, 1, (50,), dtype=bool),
    })

def create_performance_action_space():
    """Create action space for performance testing"""
    return gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

def create_performance_trajectory(episode_length, traj_id):
    """Create trajectory for performance testing"""
    observations = {
        'image': {
            'base_0_rgb': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(episode_length + 1)],
            'left_wrist_0_rgb': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(episode_length + 1)],
        },
        'image_mask': {
            'base_0_rgb': [np.array([True]) for _ in range(episode_length + 1)],
            'left_wrist_0_rgb': [np.array([True]) for _ in range(episode_length + 1)],
        },
        'state': [np.random.uniform(-1, 1, (7,)).astype(np.float32) for _ in range(episode_length + 1)],
        'tokenized_prompt': [np.random.randint(0, 1000, (50,), dtype=np.int32) for _ in range(episode_length + 1)],
        'tokenized_prompt_mask': [np.random.choice([True, False], (50,)) for _ in range(episode_length + 1)],
    }
    
    actions = [np.random.uniform(-1, 1, (7,)).astype(np.float32) for _ in range(episode_length)]
    rewards = [np.random.uniform(0, 1) for _ in range(episode_length)]
    masks = [1.0 for _ in range(episode_length - 1)] + [0.0]
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'episode_length': episode_length,
        'is_success': np.random.choice([True, False]),
        'episode_return': sum(rewards),
        'images': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(episode_length + 1)],
        'env_steps': episode_length,
        'traj_id': traj_id
    }

def test_insertion_performance():
    """Test trajectory insertion performance"""
    print("Testing Insertion Performance")
    print("=" * 40)
    
    obs_space = create_performance_obs_space()
    action_space = create_performance_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=10000)
    
    # Test different trajectory lengths
    traj_lengths = [10, 50, 100, 200]
    num_trajs = 100
    
    for length in traj_lengths:
        print(f"\nTesting with trajectory length: {length}")
        
        # Warm up
        for i in range(5):
            traj = create_performance_trajectory(episode_length=length, traj_id=i)
            buffer.insert_traj(traj)
        
        # Clear buffer for actual test
        buffer = TrajReplayBuffer(obs_space, action_space, capacity=10000)
        
        # Time insertion
        start_time = time.time()
        for i in range(num_trajs):
            traj = create_performance_trajectory(episode_length=length, traj_id=i)
            buffer.insert_traj(traj)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_traj = total_time / num_trajs
        trajs_per_second = num_trajs / total_time
        
        print(f"  Inserted {num_trajs} trajectories in {total_time:.3f}s")
        print(f"  Average time per trajectory: {avg_time_per_traj:.6f}s")
        print(f"  Trajectories per second: {trajs_per_second:.1f}")
        print(f"  Final buffer size: {buffer.size}")

def test_sampling_performance():
    """Test sampling performance"""
    print("\nTesting Sampling Performance")
    print("=" * 40)
    
    obs_space = create_performance_obs_space()
    action_space = create_performance_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=10000)
    
    # Fill buffer with data
    print("Filling buffer with data...")
    for i in range(500):  # 500 trajectories
        traj = create_performance_trajectory(episode_length=20, traj_id=i)
        buffer.insert_traj(traj)
    
    print(f"Buffer filled with {buffer.size} steps from {buffer._traj_counter} trajectories")
    
    # Test step sampling
    print("\nTesting step sampling...")
    batch_sizes = [32, 64, 128, 256, 512]
    num_samples = 1000
    
    for batch_size in batch_sizes:
        start_time = time.time()
        for _ in range(num_samples):
            sampled = buffer.sample(batch_size=batch_size)
        end_time = time.time()
        
        total_time = end_time - start_time
        samples_per_second = num_samples / total_time
        
        print(f"  Batch size {batch_size}: {samples_per_second:.1f} samples/sec")
    
    # Test trajectory sampling
    print("\nTesting trajectory sampling...")
    traj_counts = [1, 5, 10, 20, 50]
    
    for num_trajs in traj_counts:
        start_time = time.time()
        for _ in range(100):  # 100 iterations
            sampled = buffer.get_random_trajs(num_trajs=num_trajs)
        end_time = time.time()
        
        total_time = end_time - start_time
        samples_per_second = 100 / total_time
        
        print(f"  {num_trajs} trajectories: {samples_per_second:.1f} samples/sec")

def test_memory_usage():
    """Test memory usage"""
    print("\nTesting Memory Usage")
    print("=" * 40)
    
    import psutil
    import os
    
    obs_space = create_performance_obs_space()
    action_space = create_performance_action_space()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Test different buffer sizes
    capacities = [1000, 5000, 10000, 20000]
    
    for capacity in capacities:
        buffer = TrajReplayBuffer(obs_space, action_space, capacity=capacity)
        
        # Fill buffer to 80% capacity
        fill_size = int(capacity * 0.8)
        traj_length = 20
        num_trajs = fill_size // traj_length
        
        for i in range(num_trajs):
            traj = create_performance_trajectory(episode_length=traj_length, traj_id=i)
            buffer.insert_traj(traj)
        
        # Measure memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - initial_memory
        
        print(f"Capacity {capacity}: {memory_used:.1f} MB used")
        print(f"  Buffer size: {buffer.size}")
        print(f"  Memory per step: {memory_used / buffer.size:.3f} MB")
        
        # Clean up
        del buffer

def test_capacity_overflow_performance():
    """Test capacity overflow performance"""
    print("\nTesting Capacity Overflow Performance")
    print("=" * 40)
    
    obs_space = create_performance_obs_space()
    action_space = create_performance_action_space()
    buffer = TrajReplayBuffer(obs_space, action_space, capacity=1000)  # Small capacity
    
    # Insert many trajectories to trigger overflow
    num_trajs = 200
    traj_length = 10
    
    print(f"Inserting {num_trajs} trajectories of length {traj_length} into capacity {buffer.capacity}")
    
    start_time = time.time()
    for i in range(num_trajs):
        traj = create_performance_trajectory(episode_length=traj_length, traj_id=i)
        buffer.insert_traj(traj)
        
        if i % 50 == 0:
            print(f"  Inserted {i} trajectories, buffer size: {buffer.size}")
    
    end_time = time.time()
    
    total_time = end_time - start_time
    trajs_per_second = num_trajs / total_time
    
    print(f"\nCompleted in {total_time:.3f}s")
    print(f"Trajectories per second: {trajs_per_second:.1f}")
    print(f"Final buffer size: {buffer.size}/{buffer.capacity}")
    print(f"Final trajectory count: {buffer._traj_counter}")

def main():
    """Run performance tests"""
    print("TrajReplayBuffer Performance Tests")
    print("=" * 50)
    
    try:
        test_insertion_performance()
        test_sampling_performance()
        test_memory_usage()
        test_capacity_overflow_performance()
        
        print(f"\n🎉 All performance tests completed!")
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
