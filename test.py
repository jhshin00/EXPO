import jax, jax.numpy as jnp
from jax import pmap, lax
x = jnp.ones((jax.local_device_count(),), jnp.float32)
y = pmap(lambda v: lax.psum(v, 'i'), axis_name='i')(x)
print(y)  # 여기서 터지면 환경/버전 문제