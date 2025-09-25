#!/usr/bin/env python3
"""
간단한 EXPOPi0Learner 테스트
- 빠른 검증용
"""

import jax
import jax.numpy as jnp
from expo.agents.expo_pi0_learner import EXPOPi0Learner
from openpi.models import model as _model


def quick_test():
    """빠른 테스트"""
    print("🚀 Quick EXPOPi0Learner test...")
    
    try:
        # 1. 에이전트 생성
        print("Creating agent...")
        agent = EXPOPi0Learner.create(
            seed=42,
            N=2,                    # 작은 앙상블
            n_edit_samples=1,       # 편집 샘플 1개
            action_dim=7,           # 7-DOF 로봇
            action_horizon=10,      # 짧은 시퀀스 (테스트용)
            max_token_len=20,       # 짧은 토큰 (테스트용)
        )
        print(f"✅ Agent created! Step: {agent.actor.step}")
        print(f"   - Action dim: {agent.action_dim}")
        print(f"   - Action horizon: {agent.action_horizon}")
        print(f"   - T: {agent.T}")
        print(f"   - N: {agent.N}")
        
        # 2. 가짜 데이터 생성
        print("Creating fake data...")
        obs = _model.Observation(
            images={
                "base_0_rgb": jnp.ones((1, 224, 224, 3)),
                "left_wrist_0_rgb": jnp.ones((1, 224, 224, 3)),
                "right_wrist_0_rgb": jnp.ones((1, 224, 224, 3)),
            },
            image_masks={
                "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            },
            tokenized_prompt=jnp.ones((1, 10), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((1, 10), dtype=jnp.bool_),
            state=jnp.ones((1, 7)),
        )
        actions = jnp.ones((1, 10, 7))
        print("✅ Fake data created!")
        
        # 3. 액션 샘플링
        print("Testing action sampling...")
        sampled_actions = agent.sample_batch_actions(obs)
        print(f"✅ Action sampled! Shape: {sampled_actions.shape}")
        
        # 4. 간단한 학습
        print("Testing training...")
        batch = (obs, actions, jnp.array([0.5]), obs)  # (obs, actions, rewards, next_obs)
        
        # 크리틱 업데이트
        new_agent, info = agent.update_critic(batch)
        print(f"✅ Critic updated! Loss: {info['critic_loss']:.4f}")
        
        # 액터 업데이트
        actor_batch = (obs, actions)
        new_agent, info = new_agent.update_actor(actor_batch)
        print(f"✅ Actor updated! Loss: {info['actor_loss']:.4f}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    quick_test()
