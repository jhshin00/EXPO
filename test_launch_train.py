#!/usr/bin/env python3
"""
Test script for launch_train.py to verify all arguments work correctly
"""

import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, '/ssd2/EXPO')

def test_launch_train_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from expo.train.launch_train import main_app, FLAGS
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_flags_definition():
    """Test that all flags are properly defined"""
    print("\nTesting flag definitions...")
    try:
        from expo.train.launch_train import FLAGS
        
        # Test that all expected flags exist
        expected_flags = [
            'seed', 'batch_size', 'launch_group_id', 'prefix', 'suffix', 
            'wandb_project', 'wandb_entity', 'action_dim', 'action_horizon', 
            'max_token_len', 'actor_lr', 'edit_actor_lr', 'critic_lr', 
            'temp_lr', 'hidden_dims', 'discount', 'tau', 'num_qs', 
            'num_min_qs', 'critic_dropout_rate', 'critic_weight_decay',
            'critic_layer_norm', 'target_entropy', 'entropy_scale',
            'init_temperature', 'backup_entropy', 'use_pnorm',
            'adjust_target_entropy', 'actor_drop', 'N', 'T', 'n_edit_samples',
            'edit_action_scale', 'max_steps', 'start_updates', 'num_update_steps',
            'capacity', 'perform_control_evals', 'eval_episodes', 'log_interval',
            'eval_interval', 'offline_eval_interval', 'checkpoint_interval',
            'utd_ratio', 'binary_include_bc', 'max_timesteps', 'env_max_reward',
            'outputdir'
        ]
        
        missing_flags = []
        for flag_name in expected_flags:
            if not hasattr(FLAGS, flag_name):
                missing_flags.append(flag_name)
        
        if missing_flags:
            print(f"✗ Missing flags: {missing_flags}")
            return False
        else:
            print(f"✓ All {len(expected_flags)} flags are defined")
            return True
            
    except Exception as e:
        print(f"✗ Flag definition test failed: {e}")
        return False

def test_flag_values():
    """Test that flag values are accessible"""
    print("\nTesting flag values...")
    try:
        from expo.train.launch_train import FLAGS
        
        # Test some key flag values
        test_cases = [
            ('seed', 42),
            ('batch_size', 32),
            ('action_dim', 32),
            ('action_horizon', 50),
            ('max_token_len', 48),
            ('max_steps', 1000000),
            ('capacity', 1000000),
            ('perform_control_evals', True),
            ('critic_layer_norm', True),
            ('use_pnorm', False),
        ]
        
        for flag_name, expected_value in test_cases:
            actual_value = getattr(FLAGS, flag_name)
            if actual_value != expected_value:
                print(f"✗ Flag {flag_name}: expected {expected_value}, got {actual_value}")
                return False
        
        print("✓ All flag values are correct")
        return True
        
    except Exception as e:
        print(f"✗ Flag value test failed: {e}")
        return False

def test_config_creation():
    """Test that config dictionary is created correctly"""
    print("\nTesting config creation...")
    try:
        from expo.train.launch_train import main_app, FLAGS
        
        # Mock the main function to avoid actual training
        with patch('expo.train.launch_train.main') as mock_main:
            mock_main.return_value = None
            
            # Test that main_app can be called without errors
            main_app(None)
            
            # Check that main was called with a config dictionary
            assert mock_main.called, "main function should be called"
            call_args = mock_main.call_args[0]
            config = call_args[0]
            
            # Verify config is a dictionary
            assert isinstance(config, dict), "Config should be a dictionary"
            
            # Check some key config values
            expected_keys = [
                'seed', 'batch_size', 'action_dim', 'action_horizon', 
                'max_token_len', 'max_steps', 'capacity'
            ]
            
            for key in expected_keys:
                assert key in config, f"Config should contain {key}"
            
            print("✓ Config creation works correctly")
            return True
            
    except Exception as e:
        print(f"✗ Config creation test failed: {e}")
        return False

def test_main_function_compatibility():
    """Test that the main function can be called with the config"""
    print("\nTesting main function compatibility...")
    try:
        from expo.train.launch_train import FLAGS
        from expo.train.train_expo_pi0_libero import main
        
        # Create a minimal config for testing
        config = FLAGS.flag_values_dict()
        
        # Set environment variable for output directory
        os.environ['EXP'] = tempfile.mkdtemp()
        
        # Mock the training loop to avoid actual training
        with patch('expo.train.train_expo_pi0_libero.trajwise_alternating_training_loop') as mock_training:
            mock_training.return_value = None
            
            # Mock the agent creation
            with patch('expo.train.train_expo_pi0_libero.EXPOPi0Learner.create') as mock_agent:
                mock_agent.return_value = MagicMock()
                
                # Mock the replay buffer
                with patch('expo.train.train_expo_pi0_libero.TrajReplayBuffer') as mock_buffer:
                    mock_buffer.return_value = MagicMock()
                    
                    # Mock the wandb logger
                    with patch('expo.train.train_expo_pi0_libero.WandBLogger') as mock_wandb:
                        mock_wandb.return_value = MagicMock()
                        
                        # This should not raise an exception
                        main(config)
                        
                        print("✓ Main function compatibility test passed")
                        return True
                        
    except Exception as e:
        print(f"✗ Main function compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_action_spaces():
    """Test that observation and action spaces are created correctly"""
    print("\nTesting observation and action spaces...")
    try:
        from expo.train.train_expo_pi0_libero import create_expo_obs_space, create_expo_action_space
        
        # Test observation space creation
        obs_space = create_expo_obs_space()
        assert obs_space is not None, "Observation space should be created"
        assert hasattr(obs_space, 'spaces'), "Observation space should have spaces attribute"
        assert 'image' in obs_space.spaces, "Observation space should have 'image' key"
        assert 'state' in obs_space.spaces, "Observation space should have 'state' key"
        
        # Test action space creation
        action_space = create_expo_action_space()
        assert action_space is not None, "Action space should be created"
        assert hasattr(action_space, 'shape'), "Action space should have shape attribute"
        assert action_space.shape == (7,), f"Action space shape should be (7,), got {action_space.shape}"
        
        print("✓ Observation and action spaces created correctly")
        return True
        
    except Exception as e:
        print(f"✗ Observation/action space test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Testing launch_train.py")
    print("=" * 80)
    
    tests = [
        test_launch_train_imports,
        test_flags_definition,
        test_flag_values,
        test_config_creation,
        test_main_function_compatibility,
        test_observation_action_spaces,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 All tests passed! launch_train.py is working correctly!")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
