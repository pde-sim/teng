import flax.linen as nn
import jax.numpy as jnp


class PeriodicLinear2(nn.Module):
    nodes: int
    period: float

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        m = self.nodes // d
        # d, m = x.shape[-1], self.nodes
        a = self.param('a', nn.initializers.truncated_normal(1.0), (m, d))
        phi = self.param('phi', nn.initializers.truncated_normal(1.0), (m, d))
        c = self.param('c', nn.initializers.truncated_normal(1.0), (m, d))
        return (a[None, :, :] * jnp.cos((jnp.pi * 2 / self.period) * x[:, None, :] + phi[None, :, :]) + c[None, :,
                                                                                                        :]).reshape(
            x.shape[0], self.nodes)


class SimplePDENet3(nn.Module):
    width: int
    depth: int
    period: float

    @nn.compact
    def __call__(self, x):
        x = PeriodicLinear2(self.width, self.period)(x)
        x = nn.tanh(x)
        for _ in range(self.depth - 2):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)
