from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Tuple,
)

import flax
import jax
import jax.numpy as jnp
import jax.random as jrnd

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any


@dataclass(frozen=True)
class RandomNaturalPolicyGradTDVPPure:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """
    nb_params_total: int

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """

        curr_params = var_state_pure.flatten_parameters(state['params'])

        # sampled_params = jnp.take(curr_params, params_to_take)

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params, sample):
            params = var_state_pure.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = var_state_pure.evaluate(new_state, sample[None, ...]).squeeze(0)
            return value

        jac = jax.vmap(jax.grad(net_apply_func), (None, 0))(curr_params, samples.squeeze(0))[..., params_to_take] * \
              sqrt_weights[0, ..., None]

        rewards = rewards * sqrt_weights

        update, res = jnp.linalg.lstsq(jac, rewards.squeeze(0))[:2]

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), (res,)

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class RandomNaturalPolicyGradTDVP:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = RandomNaturalPolicyGradTDVPPure(self.nb_params_total)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver,
                               self.params_to_take)
