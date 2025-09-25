#!/usr/bin/env python3
"""
Simple test for launch_train.py
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/ssd2/EXPO')

def test_basic_functionality():
    """Test basic functionality without running training"""
    print("Testing launch_train.py basic functionality...")
    
    try:
        # Test imports
        from expo.train.launch_train import FLAGS, main_app
        print("✓ Imports successful")
        
        # Test flag access
        print(f"  Seed: {FLAGS.seed}")
        print(f"  Batch size: {FLAGS.batch_size}")
        print(f"  Action dim: {FLAGS.action_dim}")
        print(f"  Action horizon: {FLAGS.action_horizon}")
        print(f"  Max steps: {FLAGS.max_steps}")
        print(f"  Capacity: {FLAGS.capacity}")
        print("✓ Flag access successful")
        
        # Test config creation
        config = FLAGS.flag_values_dict()
        print(f"  Config type: {type(config)}")
        print(f"  Config keys: {len(config)}")
        print("✓ Config creation successful")
        
        # Test observation/action space creation
        from expo.train.train_expo_pi0_libero import create_expo_obs_space, create_expo_action_space
        
        obs_space = create_expo_obs_space()
        action_space = create_expo_action_space()
        
        print(f"  Observation space: {type(obs_space)}")
        print(f"  Action space: {type(action_space)}")
        print(f"  Action space shape: {action_space.shape}")
        print("✓ Space creation successful")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
