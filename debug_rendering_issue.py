#!/usr/bin/env python3
"""
렌더링 이슈를 디버깅하고 해결하는 스크립트
"""

import os
import sys
import numpy as np
import imageio
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def test_different_resolutions():
    """다양한 해상도에서 렌더링 품질을 테스트합니다."""
    
    print("🔍 Testing rendering quality at different resolutions...")
    
    # Get a simple task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(0)
    
    resolutions = [128, 256, 512]
    
    for resolution in resolutions:
        print(f"\n📐 Testing resolution: {resolution}x{resolution}")
        
        # Set environment variables
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_GL_MSAA"] = "0"
        
        try:
            # Create environment
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_args = {
                "bddl_file_name": task_bddl_file, 
                "camera_heights": resolution, 
                "camera_widths": resolution,
                "has_offscreen_renderer": True,
                "render_gpu_device_id": 0,
            }
            
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            
            # Reset and get initial observation
            env.reset()
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            
            # Check image quality
            if "agentview_image" in obs:
                img = obs["agentview_image"]
                print(f"  Image shape: {img.shape}")
                print(f"  Image dtype: {img.dtype}")
                print(f"  Image range: [{img.min()}, {img.max()}]")
                print(f"  Image std: {img.std():.2f}")
                
                # Check for artifacts
                if np.all(img == 0):
                    print(f"  ❌ All zeros detected!")
                elif img.std() < 1.0:
                    print(f"  ⚠️  Very low variance: {img.std():.2f}")
                else:
                    print(f"  ✅ Looks good")
                
                # Save sample image
                sample_path = f"debug_resolution_{resolution}.png"
                Image.fromarray(img).save(sample_path)
                print(f"  💾 Saved sample: {sample_path}")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ Error at resolution {resolution}: {e}")

def test_environment_recreation():
    """환경을 여러 번 재생성하면서 렌더링 품질을 확인합니다."""
    
    print("\n🔄 Testing environment recreation...")
    
    # Get a simple task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(0)
    
    for i in range(3):
        print(f"\n🔄 Recreation attempt {i+1}")
        
        # Set environment variables
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_GL_MSAA"] = "0"
        
        try:
            # Create environment
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_args = {
                "bddl_file_name": task_bddl_file, 
                "camera_heights": 256, 
                "camera_widths": 256,
                "has_offscreen_renderer": True,
                "render_gpu_device_id": 0,
            }
            
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            
            # Reset and get observation
            env.reset()
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            
            # Check image quality
            if "agentview_image" in obs:
                img = obs["agentview_image"]
                print(f"  Image shape: {img.shape}")
                print(f"  Image std: {img.std():.2f}")
                
                if np.all(img == 0):
                    print(f"  ❌ All zeros detected!")
                elif img.std() < 1.0:
                    print(f"  ⚠️  Very low variance: {img.std():.2f}")
                else:
                    print(f"  ✅ Looks good")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ Error in attempt {i+1}: {e}")

def test_mujoco_backend():
    """다양한 MuJoCo 백엔드를 테스트합니다."""
    
    print("\n🔧 Testing different MuJoCo backends...")
    
    backends = ["egl", "osmesa", "glfw"]
    
    for backend in backends:
        print(f"\n🔧 Testing backend: {backend}")
        
        # Set environment variables
        os.environ["MUJOCO_GL"] = backend
        os.environ["MUJOCO_GL_MSAA"] = "0"
        
        try:
            # Get a simple task
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict["libero_10"]()
            task = task_suite.get_task(0)
            
            # Create environment
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_args = {
                "bddl_file_name": task_bddl_file, 
                "camera_heights": 256, 
                "camera_widths": 256,
                "has_offscreen_renderer": True,
                "render_gpu_device_id": 0,
            }
            
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            
            # Reset and get observation
            env.reset()
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            
            # Check image quality
            if "agentview_image" in obs:
                img = obs["agentview_image"]
                print(f"  Image shape: {img.shape}")
                print(f"  Image std: {img.std():.2f}")
                
                if np.all(img == 0):
                    print(f"  ❌ All zeros detected!")
                elif img.std() < 1.0:
                    print(f"  ⚠️  Very low variance: {img.std():.2f}")
                else:
                    print(f"  ✅ Looks good")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ Error with backend {backend}: {e}")

def main():
    print("🔍 LIBERO Rendering Issue Debugger")
    print("=" * 50)
    
    # Test 1: Different resolutions
    test_different_resolutions()
    
    # Test 2: Environment recreation
    test_environment_recreation()
    
    # Test 3: Different backends
    test_mujoco_backend()
    
    print("\n✅ Debugging complete!")

if __name__ == "__main__":
    main()

