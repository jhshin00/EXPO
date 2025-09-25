#!/usr/bin/env python3
"""
단일 trajectory에 대해서만 regenerate하고 rendering해서 비디오를 저장하는 디버깅 스크립트
"""

import argparse
import json
import os
import time
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np
import imageio
import robosuite.utils.transform_utils as T
from PIL import Image
from libero.libero import benchmark

from expo_pi0.data.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def create_video_from_images(images, output_path, fps=30, camera_name="unknown"):
    """이미지 리스트로부터 비디오 생성"""
    print(f"Creating {camera_name} video: {output_path}")
    print(f"  - Frames: {len(images)}")
    print(f"  - Shape: {images[0].shape if images else 'No images'}")
    
    if not images:
        print(f"❌ No images to create video")
        return False
    
    # 비디오 생성
    imageio.mimwrite(output_path, images, fps=fps)
    
    # 첫 번째와 마지막 프레임을 이미지로도 저장
    first_frame_path = output_path.replace('.mp4', '_first_frame.png')
    last_frame_path = output_path.replace('.mp4', '_last_frame.png')
    
    # Image.fromarray(images[0]).save(first_frame_path)
    # Image.fromarray(images[-1]).save(last_frame_path)
    
    print(f"✅ Video saved: {output_path}")
    # print(f"✅ First frame: {first_frame_path}")
    # print(f"✅ Last frame: {last_frame_path}")
    
    return True


def debug_single_trajectory(args):
    """단일 trajectory를 regenerate하고 디버깅 정보를 출력"""
    
    print(f"🔍 Debugging single trajectory from {args.libero_task_suite}")
    print(f"📁 Raw data: {args.libero_raw_data_dir}")
    print(f"🎯 Task ID: {args.task_id}")
    print(f"📹 Demo ID: {args.demo_id}")
    print("=" * 80)
    
    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    
    if args.task_id >= task_suite.n_tasks:
        print(f"❌ Task ID {args.task_id} is out of range. Available tasks: 0-{task_suite.n_tasks-1}")
        return
    
    # Get specific task
    task = task_suite.get_task(args.task_id)
    print(f"📋 Task: {task.name}")
    print(f"📝 Description: {task.language}")
    
    # Get environment
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
    print(f"🌍 Environment created with resolution: {IMAGE_RESOLUTION}")
    
    # Get dataset for task
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    if not os.path.exists(orig_data_path):
        print(f"❌ Cannot find raw data file: {orig_data_path}")
        return
    
    print(f"📂 Loading original data: {orig_data_path}")
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    
    # Check available demos
    available_demos = [key for key in orig_data.keys() if key.startswith('demo_')]
    print(f"📊 Available demos: {available_demos}")
    
    if f"demo_{args.demo_id}" not in orig_data:
        print(f"❌ Demo {args.demo_id} not found in available demos")
        orig_data_file.close()
        return
    
    # Get demo data
    demo_data = orig_data[f"demo_{args.demo_id}"]
    orig_actions = demo_data["actions"][()]
    orig_states = demo_data["states"][()]
    
    print(f"📈 Original demo stats:")
    print(f"  - Actions: {len(orig_actions)}")
    print(f"  - States: {len(orig_states)}")
    print(f"  - Action shape: {orig_actions.shape}")
    print(f"  - State shape: {orig_states.shape}")
    
    # Reset environment and set initial state
    print(f"\n🔄 Resetting environment...")
    env.reset()
    env.set_init_state(orig_states[0])
    
    # Wait for environment to settle
    print(f"⏳ Settling environment...")
    for i in range(10):
        obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
        if i == 0:
            print(f"  - Initial obs keys: {list(obs.keys())}")
            print(f"  - Initial obs shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
    
    # Data collection lists
    states = []
    actions = []
    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []
    agentview_images = []
    eye_in_hand_images = []
    
    # Track no-op actions
    num_noops = 0
    noop_indices = []
    
    print(f"\n🎬 Replaying actions...")
    
    # Replay original demo actions
    for step_idx, action in enumerate(orig_actions):
        print(f"\n--- Step {step_idx} ---")
        print(f"Action: {action}")
        
        # Check for no-op
        prev_action = actions[-1] if len(actions) > 0 else None
        is_noop_action = is_noop(action, prev_action)
        
        if is_noop_action:
            print(f"⏭️  Skipping no-op action")
            num_noops += 1
            noop_indices.append(step_idx)
            continue
        
        # Record state
        if states == []:
            # First timestep - use original initial state
            states.append(orig_states[0])
            robot_states.append(demo_data["robot_states"][0])
            print(f"📊 Using original initial state")
        else:
            # Get state from environment
            current_state = env.sim.get_state().flatten()
            states.append(current_state)
            robot_states.append(
                np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
            )
            print(f"📊 Current state shape: {current_state.shape}")
        
        # Record action
        actions.append(action)
        
        # Record observations
        if "robot0_gripper_qpos" in obs:
            gripper_states.append(obs["robot0_gripper_qpos"])
        joint_states.append(obs["robot0_joint_pos"])
        ee_states.append(
            np.hstack((
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
            ))
        )
        
        # Record images
        agentview_img = obs["agentview_image"][::-1, ::-1, :].copy()
        eye_in_hand_img = obs["robot0_eye_in_hand_image"].copy()
        
        print(f"📸 Agent view image: {agentview_img.shape}, dtype: {agentview_img.dtype}")
        print(f"📸 Eye in hand image: {eye_in_hand_img.shape}, dtype: {eye_in_hand_img.dtype}")
        
        # Check for rendering issues
        if np.all(agentview_img == 0):
            print(f"⚠️  WARNING: Agent view image is all zeros!")
        if np.all(eye_in_hand_img == 0):
            print(f"⚠️  WARNING: Eye in hand image is all zeros!")
        
        agentview_images.append(agentview_img)
        eye_in_hand_images.append(eye_in_hand_img)
        
        # Execute action
        print(f"🎮 Executing action...")
        obs, reward, done, info = env.step(action.tolist())
        
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        print(f"  - Info keys: {list(info.keys()) if info else 'None'}")
    
    print(f"\n📊 Replay Summary:")
    print(f"  - Total original actions: {len(orig_actions)}")
    print(f"  - Actions after no-op filtering: {len(actions)}")
    print(f"  - No-op actions filtered: {num_noops}")
    print(f"  - No-op indices: {noop_indices}")
    print(f"  - Agent view images: {len(agentview_images)}")
    print(f"  - Eye in hand images: {len(eye_in_hand_images)}")
    
    # Create output directory
    output_dir = f"/ssd2/EXPO/debug_trajectory_{args.task_id}_{args.demo_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create videos
    print(f"\n🎥 Creating videos...")
    
    # Agent view video
    agentview_video_path = os.path.join(output_dir, "agentview_debug.mp4")
    success1 = create_video_from_images(
        agentview_images, 
        agentview_video_path, 
        fps=30, 
        camera_name="Agent View"
    )
    
    # Eye in hand video
    eye_in_hand_video_path = os.path.join(output_dir, "eye_in_hand_debug.mp4")
    success2 = create_video_from_images(
        eye_in_hand_images, 
        eye_in_hand_video_path, 
        fps=30, 
        camera_name="Eye in Hand"
    )
    
    # Side-by-side video
    if len(agentview_images) == len(eye_in_hand_images):
        side_by_side_images = []
        for i in range(len(agentview_images)):
            # Rotate agentview image 180 degrees (as done in original code)
            agentview_rotated = agentview_images[i]
            eye_in_hand_frame = eye_in_hand_images[i]
            
            # Concatenate horizontally
            combined_frame = np.concatenate([agentview_rotated, eye_in_hand_frame], axis=1)
            side_by_side_images.append(combined_frame)
        
        side_by_side_video_path = os.path.join(output_dir, "side_by_side_debug.mp4")
        success3 = create_video_from_images(
            side_by_side_images, 
            side_by_side_video_path, 
            fps=30, 
            camera_name="Side by Side"
        )
    else:
        print(f"⚠️  Cannot create side-by-side video: different frame counts")
        success3 = False
    
    # Save debug information
    debug_info = {
        "task_name": task.name,
        "task_description": task.language,
        "demo_id": args.demo_id,
        "original_actions": len(orig_actions),
        "filtered_actions": len(actions),
        "noop_count": num_noops,
        "noop_indices": noop_indices,
        "agentview_frames": len(agentview_images),
        "eye_in_hand_frames": len(eye_in_hand_images),
        "video_success": {
            "agentview": success1,
            "eye_in_hand": success2,
            "side_by_side": success3
        }
    }
    
    debug_json_path = os.path.join(output_dir, "debug_info.json")
    with open(debug_json_path, "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"\n📁 Debug output saved to: {output_dir}")
    print(f"📄 Debug info: {debug_json_path}")
    
    # Close files
    orig_data_file.close()
    
    print(f"\n✅ Debug complete!")
    print(f"🎥 Videos created:")
    print(f"  - Agent view: {'✅' if success1 else '❌'}")
    print(f"  - Eye in hand: {'✅' if success2 else '❌'}")
    print(f"  - Side by side: {'✅' if success3 else '❌'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug single trajectory regeneration")
    parser.add_argument("--libero_task_suite", type=str, default="libero_goal",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                       help="LIBERO task suite", required=True,
                       )
    parser.add_argument("--libero_raw_data_dir", type=str, default="/ssd2/EXPO/datasets/libero/libero_goal",
                       help="Path to directory containing raw HDF5 dataset", required=True,
                        )
    parser.add_argument("--task_id", type=int, default=0,
                       help="Task ID to debug (default: 0)",
                       )
    parser.add_argument("--demo_id", type=int, default=0,
                       help="Demo ID to debug (default: 0)",
                       )
    
    args = parser.parse_args()
    debug_single_trajectory(args)
