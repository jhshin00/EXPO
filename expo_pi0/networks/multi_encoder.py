from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
import flax.linen as nn

from openpi.models import model as _model

default_init = nn.initializers.xavier_uniform

PALIGEMMA_VOCAB_SIZE = 257_152

# =====================
# Utils
# =====================
def masked_mean(x: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
    """x:[B,T,D], mask:[B,T] or None"""
    if mask is None:
        return x.mean(axis=1)
    w = mask[..., None]
    denom = jnp.clip(w.sum(axis=1), min=1e-6)
    return (x * w).sum(axis=1) / denom

def sinusoidal_positional_encoding(T: int, D: int) -> jnp.ndarray:
    """[T, D] sinusoidal PE (Transformer 스타일). D는 짝수 권장."""
    position = jnp.arange(T)[:, None]
    div_term = jnp.exp(jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D))
    pe = jnp.zeros((T, D), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

# =====================
# Encoders
# =====================
class ImageEncoder(nn.Module):
    out_dim: int = 512
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = nn.Conv(32, (5,5), (2,2), padding="SAME")(x); x = nn.gelu(x)
        x = nn.Conv(64, (3,3), (2,2), padding="SAME")(x); x = nn.gelu(x)
        x = nn.Conv(128,(3,3), (2,2), padding="SAME")(x); x = nn.gelu(x)
        x = nn.Conv(256,(3,3), (2,2), padding="SAME")(x); x = nn.gelu(x)
        x = x.mean(axis=(1,2))            # [B, 256]
        x = nn.Dense(self.out_dim)(x)     # [B, out_dim]
        return x

class ViewPooler(nn.Module):
    @nn.compact
    def __call__(self, feats: jnp.ndarray) -> jnp.ndarray:   # [B,V,D] -> [B,D]
        return feats.mean(axis=1)


class LanguageEncoder(nn.Module):
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    embed_dim: int = 512
    proj_dim: Optional[int] = None
    use_positional_encoding: bool = True
    @nn.compact
    def __call__(self, token_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # token_ids: [B,T]
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(token_ids)  # [B,T,D]
        if self.use_positional_encoding:
            _, T, D = x.shape
            pe = sinusoidal_positional_encoding(T, D)  # [T,D]
            x = x + pe[None, ...]
        x = nn.LayerNorm()(x)
        x = masked_mean(x, mask)  # [B,D]
        if self.proj_dim is not None and self.proj_dim != self.embed_dim:
            x = nn.Dense(self.proj_dim)(x)
        return x


class StateEncoder(nn.Module):
    embed_dim: int = 256
    hidden: int = 256
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(x); x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim)(x)
        return x

class ActionEncoder1D(nn.Module):
    """naive한 시퀀스 인코더: pos enc + 1D conv stack + mean pool."""
    d_model: int = 256
    n_layers: int = 2
    kernel_size: int = 3
    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        # actions: [B, H, A] or [B, A] (for single timestep)
        if actions.ndim == 2:
            # Single timestep: [B, A] -> [B, 1, A]
            B, A = actions.shape
            H = 1
            actions = actions[:, None, :]  # Add horizon dimension
        else:
            # Full sequence: [B, H, A]
            B, H, A = actions.shape
        x = nn.Dense(self.d_model)(actions)
        # Positional enc along H
        pe = sinusoidal_positional_encoding(H, self.d_model)
        x = x + pe[None, ...]  # [B,H,D]
        for _ in range(self.n_layers):
            # Depthwise separable-ish: Conv1D over time dimension
            x_res = x
            x = nn.Conv(self.d_model, (self.kernel_size,), padding="SAME", feature_group_count=1)(x)  # treats last dim as channel, time=H
            x = nn.gelu(x)
            x = nn.LayerNorm()(x)
            x = x + x_res
        # mean pool over H
        x = x.mean(axis=1)  # [B, D]
        return x


# =====================
# Trunk & Critic
# =====================
class ObsTrunk(nn.Module):
    """obs만 처리. 이미지/텍스트/상태 -> concat -> 작은 MLP."""
    img_dim: int = 512
    txt_dim: int = 512
    state_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256)
    use_image: bool = True
    use_text: bool = True
    use_state: bool = True
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    image_keys: Optional[Sequence[str]] = None  # dict일 때 키 순서 고정
    @nn.compact
    def __call__(self, obs: _model.Observation) -> jnp.ndarray:
        feats = []

        # --- Images ---
        if self.use_image and obs.images is not None:
            img_enc = ImageEncoder(out_dim=self.img_dim)
            if isinstance(obs.images, dict):
                per_view = []
                keys = self.image_keys if self.image_keys is not None else list(obs.images.keys())
                for k in keys:
                    if k not in obs.images:
                        raise KeyError(f"images dict에 '{k}' 키가 없습니다.")
                    per_view.append(img_enc(obs.images[k]))
                views = jnp.stack(per_view, axis=1)  # [B,V,D]
                img_feat = ViewPooler()(views)       # [B,D]
            else:
                img_feat = img_enc(obs.images)
            feats.append(img_feat)

        # --- Text ---
        if self.use_text:
            if obs.tokenized_prompt is None:
                raise ValueError("ObsTrunk: tokenized_prompt가 필요합니다.")
            txt_enc = LanguageEncoder(
                vocab_size=self.vocab_size,
                embed_dim=self.txt_dim,
                proj_dim=self.txt_dim,
                use_positional_encoding=True,
            )
            txt_feat = txt_enc(obs.tokenized_prompt, getattr(obs, "tokenized_prompt_mask", None))
            feats.append(txt_feat)

        # --- State ---
        if self.use_state and obs.state is not None:
            st_enc = StateEncoder(embed_dim=self.state_dim)
            st_feat = st_enc(obs.state)
            feats.append(st_feat)

        if not feats:
            raise ValueError("ObsTrunk: 사용할 모달리티가 없습니다.")

        x = jnp.concatenate(feats, axis=-1)  # [B, sum_dims]
        # small MLP head
        for hd in self.hidden_dims:
            x = nn.Dense(hd)(x); x = nn.gelu(x)
            x = nn.LayerNorm()(x)
        return x  # [B, D_s]


class _MLP(nn.Module):
    hidden_dims: Sequence[int]
    dropout_rate: Optional[float] = None
    activate_final: bool = False
    use_layer_norm: bool = False
    use_pnorm: bool = False
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, hd in enumerate(self.hidden_dims):
            x = nn.Dense(hd)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                if self.dropout_rate:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                x = nn.gelu(x)
        if self.use_pnorm:
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
        return x

class MultiEncoder(nn.Module):
    """
    actor용 base_cls: obs + actions 둘 다 인코딩해서 feature 반환.
    - ObsTrunk(**trunk_kwargs)
    - ActionEncoder1D(**act_kwargs)
    - fuse(concat) 후 작은 MLP로 최종 feature 생성 → TanhNormal이 [H*A] 출력
    """
    # trunk_kwargs: dict
    img_dim: int = 512
    txt_dim: int = 512
    state_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256)
    use_image: bool = True
    use_text: bool = True
    use_state: bool = True
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    image_keys: Optional[Sequence[str]] = None  # dict일 때 키 순서 고정
    # act_kwargs: dict
    d_model: int = 256
    n_layers: int = 2
    kernel_size: int = 3
    # #
    dropout_rate: Optional[float] = None
    activate_final: bool = True
    use_pnorm: bool = False
    use_layer_norm: bool = False  # 필요하면 True로

    @nn.compact
    def __call__(self, observations, actions, training: bool = False):
        # 1) obs -> s_feat
        s_feat = ObsTrunk(
            img_dim=self.img_dim,
            txt_dim=self.txt_dim,
            state_dim=self.state_dim,
            hidden_dims=self.hidden_dims,
            use_image=self.use_image,
            use_text=self.use_text,
            use_state=self.use_state,
            vocab_size=self.vocab_size,
            image_keys=self.image_keys,
        )(observations)          # [B, D_s]
        # 2) act -> a_feat  (시퀀스 구조 유지)
        a_feat = ActionEncoder1D(
            d_model=self.d_model,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
        )(actions)          # [B, D_a]
        # 3) fuse & head
        h = jnp.concatenate([s_feat, a_feat], axis=-1)                          # [B, D_s + D_a]
        z = _MLP(
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activate_final=self.activate_final,
            use_layer_norm=self.use_layer_norm,
            use_pnorm=self.use_pnorm,
        )(h, training)                                                          # [B, D_z]
        return z