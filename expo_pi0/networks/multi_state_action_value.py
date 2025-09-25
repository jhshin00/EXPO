import flax.linen as nn
import jax.numpy as jnp

from expo_pi0.networks import default_init
from openpi.models import model as _model


# class MultiStateValue(nn.Module):
#     base_cls: nn.Module

#     @nn.compact
#     def __call__(
#         self, observations, *args, **kwargs
#     ) -> jnp.ndarray:
#         # observations가 딕셔너리인 경우 MultiEncoder 사용
#         if isinstance(observations, dict):
#             outputs = self.base_cls()(observations, *args, **kwargs)
#         else:
#             # 기존 로직 (tensor인 경우)
#             inputs = jnp.concatenate([observations], axis=-1)
#             outputs = self.base_cls()(inputs, *args, **kwargs)

#         value = nn.Dense(1, kernel_init=default_init())(outputs)

#         return jnp.squeeze(value, -1)


class MultiStateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: _model.Observation, actions: _model.Actions, *args, **kwargs
    ) -> jnp.ndarray:
        # Get observation features using MultiEncoder (which now handles actions)
        feat = self.base_cls()(observations, actions, *args, **kwargs)
        
        # Final Q-value computation
        value = nn.Dense(1, kernel_init=default_init())(feat)
        return jnp.squeeze(value, -1)
