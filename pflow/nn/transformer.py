import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""
  def __init__(self,num_heads,num_layers,key_size,model_size,widening_factor=1,name="transformer"):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.key_size = key_size
    self.model_size = model_size
    self.widening_factor = widening_factor
    self.initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    self.attention = hk.MultiHeadAttention(num_heads, key_size=key_size,model_size=model_size,w_init=self.initializer)
    self.dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=self.initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=self.initializer),
      ])
    self.attention_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    self.ffn_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    

  def __call__(
      self,
      x: jnp.ndarray,  # [T, D]
      t: jnp.ndarray
  ) -> jnp.ndarray:  # [T, D]
    """Transforms input embedding sequences to output embedding sequences."""
    seq_len, dim = x.shape

    # normalize the input along each dimension
    # h,center,vh = standardize_points(x)
    h,center = center_points(x)
    # h = x
    h = jnp.concatenate([h,jnp.repeat(jnp.array(t).reshape(1, 1), seq_len, axis=0)], axis = -1)
    
    for _ in range(self.num_layers):
      # First the attention block.
      h_attn = self.attention (h, h, h)
      h = h + h_attn
      # h = self.attention_norm(h)

      h_dense = self.dense_block(h)
      h = h + h_dense
      # h = self.ffn_norm(h)
    
    # result = reverse_standardize_points(hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(0.01))(h),vh,center)
    result = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(0.1))(h)
    # result = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(0.1))(h)
    
    return result
  
def center_points(x):
  # Perform centerization to data
  center = jnp.mean(x,axis=0)
  centered_x = x - center
  
  return centered_x,center
  

def standardize_points(x: jnp.ndarray) -> jnp.ndarray:
  """Applies standardization to x with default settings."""
  # Perform centerization to data
  center = jnp.mean(x,axis=0)
  centered_x = x - center
  
  # Perform SVD on the centered data matrix
  _, _, vh = jnp.linalg.svd(centered_x)
  
  # Align the centered points to the principal axes
  aligned_points = centered_x.dot(vh.T)
    
  return aligned_points,center,vh

def standardize_with(x,center,vh) -> jnp.ndarray:
  """Applies standardization to x with default settings."""
  # Perform centerization to data=
  centered_x = x - center
  
  # Align the centered points to the principal axes
  aligned_points = centered_x.dot(vh.T)
    
  return aligned_points

def reverse_standardize_points(aligned_points, vh, centroid):
    # Rotate the points back to their original orientation
    original_points = aligned_points.dot(vh)

    return original_points

def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  """Applies a unique LayerNorm to x with default settings."""
  return x

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes, model_size):
    x = jax.random.uniform(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Transformer(num_heads, num_layers, key_sizes, model_size)
        return net(x.reshape(n, dim),t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    # div_fn = lambda _params, _x, _t, _v: div(lambda _x: network.apply(_params, _x, _t))(_v, _x)
    return params, network.apply
