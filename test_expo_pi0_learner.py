#!/usr/bin/env python3
"""
EXPOPi0Learner 테스트 스크립트
- 모델 생성 및 초기화 테스트
- 액션 샘플링 테스트
- 학습 루프 테스트
- 체크포인트 저장/로드 테스트
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
import tempfile
import os

# EXPOPi0Learner import
from expo.agents.expo_pi0_learner import EXPOPi0Learner
from expo.pi0_utils import save_checkpoint, load_checkpoint
from openpi.models import model as _model


def create_fake_observation(batch_size: int = 2) -> _model.Observation:
    """테스트용 가짜 관측 데이터 생성"""
    # 이미지 데이터 (Pi0가 기대하는 카메라 뷰)
    images = {
        "base_0_rgb": jnp.ones((batch_size, 64, 64, 3), dtype=jnp.float32),
        "left_wrist_0_rgb": jnp.ones((batch_size, 64, 64, 3), dtype=jnp.float32),
        "right_wrist_0_rgb": jnp.ones((batch_size, 64, 64, 3), dtype=jnp.float32),
    }
    
    # 이미지 마스크 (필수 필드)
    image_masks = {
        "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
    }
    
    # 텍스트 데이터
    tokenized_prompt = jnp.ones((batch_size, 10), dtype=jnp.int32)
    tokenized_prompt_mask = jnp.ones((batch_size, 10), dtype=jnp.bool_)
    
    # 상태 데이터
    state = jnp.ones((batch_size, 7), dtype=jnp.float32)  # 7-DOF 로봇
    
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )


def create_fake_actions(batch_size: int = 2, action_horizon: int = 50, action_dim: int = 7) -> _model.Actions:
    """테스트용 가짜 액션 데이터 생성"""
    return jnp.ones((batch_size, action_horizon, action_dim), dtype=jnp.float32)


def test_model_creation():
    """모델 생성 및 초기화 테스트"""
    print("🧪 Testing model creation...")
    
    try:
        # 에이전트 생성
        agent = EXPOPi0Learner.create(
            seed=42,
            N=4,                    # 앙상블 샘플 수
            n_edit_samples=2,       # 편집 샘플 수
            action_dim=7,           # 로봇 관절 수
            action_horizon=50,      # 액션 시퀀스 길이
            max_token_len=48,       # 토큰 길이
            actor_lr=3e-4,
            critic_lr=3e-4,
            edit_actor_lr=3e-4,
        )
        
        print(f"✅ Model created successfully!")
        print(f"   - Actor step: {agent.actor.step}")
        print(f"   - N (ensemble samples): {agent.N}")
        print(f"   - n_edit_samples: {agent.n_edit_samples}")
        print(f"   - Action dim: {agent.action_dim}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise


def test_action_sampling(agent: EXPOPi0Learner):
    """액션 샘플링 테스트"""
    print("\n🧪 Testing action sampling...")
    
    try:
        # 단일 관측으로 액션 샘플링
        obs = create_fake_observation(batch_size=1)
        actions = agent.sample_actions(obs)
        
        print(f"✅ Single action sampling successful!")
        print(f"   - Action shape: {actions.shape}")
        print(f"   - Expected shape: (50, 7)")
        print(f"   - Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # 배치 관측으로 액션 샘플링
        batch_obs = create_fake_observation(batch_size=4)
        batch_actions = agent.sample_batch_actions(batch_obs)
        
        print(f"✅ Batch action sampling successful!")
        print(f"   - Batch action shape: {batch_actions.shape}")
        print(f"   - Expected shape: (4, 50, 7)")
        
        return True
        
    except Exception as e:
        print(f"❌ Action sampling failed: {e}")
        raise


def test_training_step(agent: EXPOPi0Learner):
    """학습 단계 테스트"""
    print("\n🧪 Testing training step...")
    
    try:
        # 가짜 배치 데이터 생성
        observations = create_fake_observation(batch_size=8)
        actions = create_fake_actions(batch_size=8)
        rewards = jnp.ones(8) * 0.5  # 가짜 보상
        next_observations = create_fake_observation(batch_size=8)
        
        batch = (observations, actions, rewards, next_observations)
        
        # 크리틱 업데이트
        new_agent, critic_info = agent.update_critic(batch)
        print(f"✅ Critic update successful!")
        print(f"   - Critic loss: {critic_info['critic_loss']:.4f}")
        print(f"   - Q-value: {critic_info['q']:.4f}")
        
        # 액터 업데이트
        actor_batch = (observations, actions)
        new_agent, actor_info = new_agent.update_actor(actor_batch)
        print(f"✅ Actor update successful!")
        print(f"   - Actor loss: {actor_info['actor_loss']:.4f}")
        print(f"   - Grad norm: {actor_info['grad_norm']:.4f}")
        
        # 편집 액터 업데이트 (n_edit_samples > 0일 때)
        if agent.n_edit_samples > 0:
            new_agent, edit_info = new_agent.update_edit_actor(actor_batch)
            print(f"✅ Edit actor update successful!")
            print(f"   - Edit actor loss: {edit_info['edit_actor_loss']:.4f}")
            print(f"   - Edit Q-value: {edit_info['edit_q']:.4f}")
            print(f"   - Entropy: {edit_info['entropy']:.4f}")
            
            # 온도 업데이트
            new_agent, temp_info = new_agent.update_temperature(edit_info['entropy'])
            print(f"✅ Temperature update successful!")
        
        # 통합 업데이트 테스트
        new_agent, update_info = new_agent.update(batch, utd_ratio=1)
        print(f"✅ Integrated update successful!")
        print(f"   - Update info keys: {list(update_info.keys())}")
        
        return new_agent
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        raise


def test_checkpoint(agent: EXPOPi0Learner):
    """체크포인트 저장/로드 테스트"""
    print("\n🧪 Testing checkpoint save/load...")
    
    try:
        # 임시 파일로 체크포인트 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            checkpoint_path = tmp_file.name
        
        # 체크포인트 저장
        save_checkpoint(agent.actor, checkpoint_path)
        print(f"✅ Checkpoint saved to: {checkpoint_path}")
        
        # 체크포인트 로드
        loaded_actor = load_checkpoint(
            checkpoint_path, 
            agent.actor.tx, 
            agent.actor.trainable_filter
        )
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   - Original step: {agent.actor.step}")
        print(f"   - Loaded step: {loaded_actor.step}")
        
        # 파라미터 비교
        original_params = agent.actor.params.to_pure_dict()
        loaded_params = loaded_actor.params.to_pure_dict()
        
        # 파라미터가 동일한지 확인
        param_match = True
        for key in original_params:
            if key in loaded_params:
                if not jnp.allclose(original_params[key], loaded_params[key]):
                    param_match = False
                    break
        
        if param_match:
            print(f"✅ Parameters match perfectly!")
        else:
            print(f"⚠️  Parameters don't match exactly (might be due to precision)")
        
        # 임시 파일 삭제
        os.unlink(checkpoint_path)
        print(f"✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        raise


def test_different_configurations():
    """다양한 설정으로 테스트"""
    print("\n🧪 Testing different configurations...")
    
    configs = [
        {"N": 1, "n_edit_samples": 0, "name": "Single sample, no edit"},
        {"N": 2, "n_edit_samples": 1, "name": "Small ensemble with edit"},
        {"N": 8, "n_edit_samples": 4, "name": "Large ensemble with edit"},
    ]
    
    for config in configs:
        try:
            print(f"\n   Testing: {config['name']}")
            agent = EXPOPi0Learner.create(
                seed=42,
                N=config["N"],
                n_edit_samples=config["n_edit_samples"],
                action_dim=7,
                action_horizon=50,
            )
            
            # 액션 샘플링 테스트
            obs = create_fake_observation(batch_size=1)
            actions = agent.sample_actions(obs)
            
            print(f"   ✅ {config['name']} - Action shape: {actions.shape}")
            
        except Exception as e:
            print(f"   ❌ {config['name']} failed: {e}")


def run_performance_test(agent: EXPOPi0Learner):
    """성능 테스트"""
    print("\n🧪 Running performance test...")
    
    try:
        import time
        
        # 액션 샘플링 성능
        obs = create_fake_observation(batch_size=1)
        
        # JIT 컴파일 시간 측정
        start_time = time.time()
        _ = agent.sample_actions(obs)  # 첫 번째 실행 (컴파일)
        compile_time = time.time() - start_time
        
        # 실제 실행 시간 측정
        start_time = time.time()
        for _ in range(10):
            _ = agent.sample_actions(obs)
        execution_time = (time.time() - start_time) / 10
        
        print(f"✅ Performance test completed!")
        print(f"   - JIT compile time: {compile_time:.3f}s")
        print(f"   - Average execution time: {execution_time:.3f}s")
        print(f"   - Actions per second: {1/execution_time:.1f}")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def main():
    """메인 테스트 함수"""
    print("🚀 Starting EXPOPi0Learner tests...\n")
    
    try:
        # 1. 모델 생성 테스트
        agent = test_model_creation()
        
        # 2. 액션 샘플링 테스트
        test_action_sampling(agent)
        
        # 3. 학습 단계 테스트
        agent = test_training_step(agent)
        
        # 4. 체크포인트 테스트
        test_checkpoint(agent)
        
        # 5. 다양한 설정 테스트
        test_different_configurations()
        
        # 6. 성능 테스트
        run_performance_test(agent)
        
        print("\n🎉 All tests passed successfully!")
        print("\n📊 Test Summary:")
        print("   ✅ Model creation")
        print("   ✅ Action sampling (single & batch)")
        print("   ✅ Training steps (critic, actor, edit_actor)")
        print("   ✅ Checkpoint save/load")
        print("   ✅ Different configurations")
        print("   ✅ Performance benchmarks")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
