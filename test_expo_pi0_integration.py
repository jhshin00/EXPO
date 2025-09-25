#!/usr/bin/env python
"""Test script to verify expo_pi0_learner integration with LIBERO"""

import numpy as np
import jax.numpy as jnp
from openpi.models import model as _model
from expo.agents.expo_pi0_learner import EXPOPi0Learner

def create_fake_observation(batch_size: int = 1) -> _model.Observation:
    """Create a fake observation for testing"""
    return _model.Observation(
        images={
            "base_0_rgb": np.random.randn(batch_size, 224, 224, 3).astype(np.float32),
            "left_wrist_0_rgb": np.random.randn(batch_size, 224, 224, 3).astype(np.float32),
            "right_wrist_0_rgb": np.random.randn(batch_size, 224, 224, 3).astype(np.float32),
        },
        image_masks={
            "base_0_rgb": np.ones(batch_size, dtype=bool),
            "left_wrist_0_rgb": np.ones(batch_size, dtype=bool),
            "right_wrist_0_rgb": np.ones(batch_size, dtype=bool),
        },
        state=np.random.randn(batch_size, 8).astype(np.float32),
    )

def create_fake_actions(batch_size: int = 1, action_horizon: int = 50, action_dim: int = 8) -> _model.Actions:
    """Create fake actions for testing"""
    return np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32)

def test_expo_pi0_learner_creation():
    """Test creating expo_pi0_learner"""
    print("Testing EXPOPi0Learner creation...")
    
    try:
        agent = EXPOPi0Learner.create(
            seed=42,
            actor_lr=3e-4,
            critic_lr=3e-4,
            temp_lr=3e-4,
            hidden_dims=(256, 256),
            discount=0.99,
            tau=0.005,
            num_qs=2,
            action_dim=8,
            action_horizon=50,
            max_token_len=48,
            N=1,
            T=10,
            n_edit_samples=0,
            edit_action_scale=1.0,
        )
        print("✓ EXPOPi0Learner created successfully")
        return agent
    except Exception as e:
        print(f"✗ Failed to create EXPOPi0Learner: {e}")
        return None

def test_action_sampling(agent):
    """Test action sampling"""
    print("Testing action sampling...")
    
    try:
        obs = create_fake_observation(batch_size=1)
        action, updated_agent = agent.sample_actions(obs)
        
        print(f"✓ Action sampled successfully, shape: {action.shape}")
        print(f"✓ Expected shape: (50, 8), Got: {action.shape}")
        
        if action.shape == (50, 8):
            print("✓ Action shape is correct")
        else:
            print("✗ Action shape is incorrect")
            
        return updated_agent
    except Exception as e:
        print(f"✗ Failed to sample actions: {e}")
        return None

def test_batch_action_sampling(agent):
    """Test batch action sampling"""
    print("Testing batch action sampling...")
    
    try:
        obs = create_fake_observation(batch_size=4)
        actions, updated_agent = agent.sample_batch_actions(obs)
        
        print(f"✓ Batch actions sampled successfully, shape: {actions.shape}")
        print(f"✓ Expected shape: (4, 50, 8), Got: {actions.shape}")
        
        if actions.shape == (4, 50, 8):
            print("✓ Batch action shape is correct")
        else:
            print("✗ Batch action shape is incorrect")
            
        return updated_agent
    except Exception as e:
        print(f"✗ Failed to sample batch actions: {e}")
        return None

def test_training_step(agent):
    """Test a single training step"""
    print("Testing training step...")
    
    try:
        # Create fake batch
        obs = create_fake_observation(batch_size=4)
        actions = create_fake_actions(batch_size=4)
        rewards = np.random.randn(4).astype(np.float32)
        next_obs = create_fake_observation(batch_size=4)
        
        batch = (obs, actions, rewards, next_obs)
        
        # Update agent
        updated_agent, info = agent.update(batch, utd_ratio=1)
        
        print(f"✓ Training step completed successfully")
        print(f"✓ Update info: {info}")
        
        return updated_agent
    except Exception as e:
        print(f"✗ Failed training step: {e}")
        return None

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing expo_pi0_learner integration")
    print("=" * 50)
    
    # Test 1: Create agent
    agent = test_expo_pi0_learner_creation()
    if agent is None:
        return
    
    print()
    
    # Test 2: Sample actions
    agent = test_action_sampling(agent)
    if agent is None:
        return
    
    print()
    
    # Test 3: Sample batch actions
    agent = test_batch_action_sampling(agent)
    if agent is None:
        return
    
    print()
    
    # Test 4: Training step
    agent = test_training_step(agent)
    if agent is None:
        return
    
    print()
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

if __name__ == "__main__":
    main()





