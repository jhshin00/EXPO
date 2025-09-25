#!/usr/bin/env python3
"""
LIBERO HDF5 파일에서 이미지 데이터를 추출하여 비디오를 생성하는 스크립트
"""

import h5py
import numpy as np
import imageio
import pathlib
from PIL import Image

def create_video_from_libero_hdf5(hdf5_file, demo_idx=0, camera_type='agentview', output_video_path="libero_video.mp4", fps=30):
    """
    LIBERO HDF5 파일에서 특정 데모의 이미지 시퀀스를 비디오로 생성합니다.
    
    Args:
        hdf5_file: HDF5 파일 경로
        demo_idx: 사용할 데모 인덱스 (기본값: 0)
        camera_type: 'agentview' 또는 'eye_in_hand' (기본값: 'agentview')
        output_video_path: 출력 비디오 파일 경로
        fps: 비디오 프레임 레이트
    """
    
    print(f"Creating video from: {hdf5_file}")
    print(f"Demo index: {demo_idx}, Camera: {camera_type}")
    
    # 이미지 데이터 경로
    image_path = f'data/demo_{demo_idx}/obs/{camera_type}_rgb'
    
    with h5py.File(hdf5_file, 'r') as f:
        if image_path not in f:
            print(f"❌ Image data not found at: {image_path}")
            print("Available demos:")
            for key in f['data'].keys():
                if key.startswith('demo_'):
                    print(f"  - {key}")
            return False
        
        # 이미지 데이터 로드
        image_data = f[image_path]
        print(f"✅ Found image data: {image_data.shape}, {image_data.dtype}")
        
        # 이미지 시퀀스 추출
        images = []
        for i in range(image_data.shape[0]):
            frame = image_data[i][::-1, ::-1, :]  # (H, W, C) 형태
            images.append(frame)
        
        print(f"Extracted {len(images)} frames")
        
        # 비디오 생성
        print(f"Creating video: {output_video_path}")
        imageio.mimwrite(output_video_path, images, fps=fps)
        
        # 첫 번째와 마지막 프레임을 이미지로도 저장
        first_frame_path = output_video_path.replace('.mp4', '_first_frame.png')
        last_frame_path = output_video_path.replace('.mp4', '_last_frame.png')
        
        Image.fromarray(images[0]).save(first_frame_path)
        Image.fromarray(images[-1]).save(last_frame_path)
        
        print(f"✅ Video saved: {output_video_path}")
        print(f"✅ First frame: {first_frame_path}")
        print(f"✅ Last frame: {last_frame_path}")
        
        return True

def create_side_by_side_video(hdf5_file, demo_idx=0, output_video_path="libero_side_by_side.mp4", fps=30):
    """
    agentview와 eye_in_hand 카메라를 나란히 배치한 비디오를 생성합니다.
    """
    
    print(f"Creating side-by-side video from: {hdf5_file}")
    print(f"Demo index: {demo_idx}")
    
    agentview_path = f'data/demo_{demo_idx}/obs/agentview_rgb'
    eye_in_hand_path = f'data/demo_{demo_idx}/obs/eye_in_hand_rgb'
    
    with h5py.File(hdf5_file, 'r') as f:
        if agentview_path not in f or eye_in_hand_path not in f:
            print(f"❌ Required image data not found")
            return False
        
        agentview_data = f[agentview_path]
        eye_in_hand_data = f[eye_in_hand_path]
        
        print(f"Agent view: {agentview_data.shape}")
        print(f"Eye in hand: {eye_in_hand_data.shape}")
        
        # 두 카메라의 프레임 수가 같은지 확인
        min_frames = min(agentview_data.shape[0], eye_in_hand_data.shape[0])
        print(f"Using {min_frames} frames")
        
        # 나란히 배치된 이미지 생성
        side_by_side_images = []
        for i in range(min_frames):
            agentview_frame = agentview_data[i][::-1, ::-1, :]
            eye_in_hand_frame = eye_in_hand_data[i]
            
            # 두 이미지를 나란히 배치 (가로로 연결)
            combined_frame = np.concatenate([agentview_frame, eye_in_hand_frame], axis=1)
            side_by_side_images.append(combined_frame)
        
        # 비디오 생성
        print(f"Creating side-by-side video: {output_video_path}")
        imageio.mimwrite(output_video_path, side_by_side_images, fps=fps)
        
        print(f"✅ Side-by-side video saved: {output_video_path}")
        return True

def main():
    hdf5_file = "/ssd2/EXPO/datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"
    
    print("=" * 60)
    print("LIBERO HDF5 Video Creator")
    print("=" * 60)
    
    # 1. Agent view 비디오 생성
    print("\n1. Creating Agent View Video")
    print("-" * 40)
    success1 = create_video_from_libero_hdf5(
        hdf5_file, 
        demo_idx=0, 
        camera_type='agentview',
        output_video_path="/ssd2/EXPO/libero_agentview.mp4",
        fps=30
    )
    
    # 2. Eye in hand 비디오 생성
    print("\n2. Creating Eye in Hand Video")
    print("-" * 40)
    success2 = create_video_from_libero_hdf5(
        hdf5_file, 
        demo_idx=0, 
        camera_type='eye_in_hand',
        output_video_path="/ssd2/EXPO/libero_eye_in_hand.mp4",
        fps=30
    )
    
    # 3. 나란히 배치된 비디오 생성
    print("\n3. Creating Side-by-Side Video")
    print("-" * 40)
    success3 = create_side_by_side_video(
        hdf5_file,
        demo_idx=0,
        output_video_path="/ssd2/EXPO/libero_side_by_side.mp4",
        fps=30
    )
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Agent view video: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Eye in hand video: {'✅ Success' if success2 else '❌ Failed'}")
    print(f"Side-by-side video: {'✅ Success' if success3 else '❌ Failed'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
