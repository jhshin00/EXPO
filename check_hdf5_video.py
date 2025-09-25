#!/usr/bin/env python3
"""
HDF5 파일에서 이미지 데이터를 추출하여 비디오로 확인하는 스크립트
"""

import h5py
import numpy as np
import imageio
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

def check_hdf5_structure(file_path):
    """HDF5 파일의 구조를 확인합니다."""
    print(f"Checking HDF5 file structure: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
        
        f.visititems(print_structure)

def extract_images_and_create_video(file_path, output_video_path="output_video.mp4", fps=30):
    """HDF5 파일에서 이미지 데이터를 추출하여 비디오를 생성합니다."""
    
    print(f"Extracting images from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # LIBERO 데이터셋의 실제 이미지 데이터 경로들
        possible_image_paths = [
            'data/demo_0/obs/agentview_rgb',
            'data/demo_0/obs/eye_in_hand_rgb',
            'data/demo_1/obs/agentview_rgb',
            'data/demo_1/obs/eye_in_hand_rgb',
            'data/demo_2/obs/agentview_rgb',
            'data/demo_2/obs/eye_in_hand_rgb'
        ]
        
        image_data = None
        image_path = None
        
        # 가능한 경로들에서 이미지 데이터 찾기
        for path in possible_image_paths:
            if path in f:
                image_data = f[path]
                image_path = path
                print(f"Found image data at: {path}")
                print(f"Shape: {image_data.shape}, Dtype: {image_data.dtype}")
                break
        
        if image_data is None:
            print("No image data found. Available keys:")
            def print_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape}, {obj.dtype}")
            f.visititems(print_keys)
            return False
        
        # 이미지 데이터 추출
        images = []
        
        if len(image_data.shape) == 4:  # (T, H, W, C) 또는 (T, C, H, W)
            print(f"4D data found: {image_data.shape}")
            
            # 첫 번째 프레임 확인
            first_frame = image_data[0]
            print(f"First frame shape: {first_frame.shape}")
            
            # 데이터가 (T, C, H, W) 형태인지 (T, H, W, C) 형태인지 확인
            if first_frame.shape[0] == 3 or first_frame.shape[0] == 1:  # (C, H, W)
                print("Data format: (T, C, H, W)")
                for i in range(image_data.shape[0]):
                    frame = image_data[i]  # (C, H, W)
                    if frame.shape[0] == 3:  # RGB
                        frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
                    else:  # Grayscale
                        frame = frame[0]  # (H, W)
                    images.append(frame)
            else:  # (H, W, C)
                print("Data format: (T, H, W, C)")
                for i in range(image_data.shape[0]):
                    frame = image_data[i]  # (H, W, C)
                    images.append(frame)
                    
        elif len(image_data.shape) == 3:  # (H, W, C) 또는 (C, H, W)
            print(f"3D data found: {image_data.shape}")
            if image_data.shape[0] == 3 or image_data.shape[0] == 1:  # (C, H, W)
                frame = image_data
                if frame.shape[0] == 3:  # RGB
                    frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
                else:  # Grayscale
                    frame = frame[0]  # (H, W)
                images.append(frame)
            else:  # (H, W, C)
                images.append(image_data)
        else:
            print(f"Unexpected data shape: {image_data.shape}")
            return False
        
        print(f"Extracted {len(images)} images")
        
        if len(images) == 0:
            print("No images extracted")
            return False
        
        # 이미지 정규화 및 타입 변환
        processed_images = []
        for img in images:
            # 데이터 타입 확인 및 변환
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    # 0-1 범위의 float 데이터
                    img = (img * 255).astype(np.uint8)
                else:
                    # 다른 범위의 데이터
                    img = np.clip(img, 0, 255).astype(np.uint8)
            
            # 그레이스케일인 경우 RGB로 변환
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            
            processed_images.append(img)
        
        # 비디오 생성
        print(f"Creating video: {output_video_path}")
        print(f"Video shape: {processed_images[0].shape}")
        
        # imageio로 비디오 저장
        imageio.mimwrite(output_video_path, processed_images, fps=fps)
        
        print(f"Video saved successfully: {output_video_path}")
        
        # 첫 번째 프레임을 이미지로도 저장
        first_frame_path = output_video_path.replace('.mp4', '_first_frame.png')
        Image.fromarray(processed_images[0]).save(first_frame_path)
        print(f"First frame saved: {first_frame_path}")
        
        return True

def main():
    hdf5_file = "/ssd2/EXPO/datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"
    output_video = "/ssd2/EXPO/kitchen_scene1_drawer_video.mp4"
    
    # 1. HDF5 파일 구조 확인
    print("=" * 50)
    print("1. HDF5 File Structure")
    print("=" * 50)
    #check_hdf5_structure(hdf5_file)
    
    print("\n" + "=" * 50)
    print("2. Extracting Images and Creating Video")
    print("=" * 50)
    
    # 2. 이미지 추출 및 비디오 생성
    success = extract_images_and_create_video(hdf5_file, output_video, fps=30)
    
    if success:
        print(f"\n✅ Success! Video created: {output_video}")
    else:
        print("\n❌ Failed to create video")

if __name__ == "__main__":
    main()
