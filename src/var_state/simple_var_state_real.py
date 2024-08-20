import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from math import prod
from typing import (
    Any,
    Callable,
    Tuple,
)

import flax
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from jax import tree_util, flatten_util

from src.sampler import AbstractSampler
from src.var_state import AbstractVarState, AbstractVarStatePure

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any


@dataclass(frozen=True)  # to make it immutable and hashable
class SimpleVarStateRealPure(AbstractVarStatePure):
    """
    real valued simple var_state
    parameters must also be real valued
    """
    net: NN
    system_shape: Shape
    param_unravel_func: Callable

    def flatten_parameters(self, params):  # TODO: MAYBE CHANGE THIS TO jax.flatten_util.ravel_pytree
        """
        naive way of flatten parameters without distinguishing real and complex values
        very likely you want to use flatten_parameters instead of this method
        use with caution
        """
        return flatten_util.ravel_pytree(params)[0]

    @partial(jax.jit, static_argnums=0)
    def flatten_parameters_jitted(self, params):
        return self.flatten_parameters(params)

    def unflatten_parameters(self, params):
        """
        naive way of unflatten parameters without distinguishing real and complex values
        very likely you want to use unflatten_parameters instead of this method
        use with caution
        """
        return self.param_unravel_func(params)

    @partial(jax.jit, static_argnums=0)
    def unflatten_parameters_jitted(self, params):
        return self.unflatten_parameters(params)

    def evaluate(self, state, samples):
        return self.net.apply(state, samples)  # .real # explicitly make the output real

    def jvp_vjp_func(self, state, samples):
        """
        samples will be applied in parallel to multiple devices
        we will not jit inside this function as it can be jitted later one when needed
        """

        curr_params = self.flatten_parameters(state['params'])

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params):
            params = self.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = self.evaluate_pmapped(new_state, samples)
            return value

        def jvp_func(tangents):
            pushforwards = jax.jvp(net_apply_func, (curr_params,), (tangents,))[1]
            return pushforwards

        value, vjp_func_raw = jax.vjp(net_apply_func, curr_params)

        def vjp_func(cotangents):
            return vjp_func_raw(cotangents)[0]

        return jvp_func, vjp_func, value

    def jacobian(self, state, samples):
        """
        returns the jacobian matrix regarding the current samples
        return a complex valued matrix or array,
        with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
        """

        # commented out jit because very likely this function is used only once per compile
        # @jax.jit
        def net_apply_func(params):
            params = self.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = self.evaluate(new_state, samples)
            return value

        jac = jax.jacrev(net_apply_func)(self.flatten_parameters(state['params']))
        return jac

    # already implemented in abstract var_state. copied here for clarity
    # @partial(jax.jit, static_argnums=0)
    # def jac_log_psi_jitted(self, state, samples):
    #     return self.jac_log_psi(state, samples)

    # already implemented in abstract var_state. copied here for clarity
    # @partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=0)
    # def jac_log_psi_pmapped(self, state, samples):
    #     return self.jac_log_psi(state, samples)


class SimpleVarStateReal(AbstractVarState):
    """
    real valued simple var_state
    parameters must also be real valued
    """

    def __init__(self, net, system_shape, sampler, init_seed=1234, init_sample=None):
        """
        batch_size: the batch_size used for sampling
        sampler: a sampler that samples from (normalized) |psi|^2
        init_seed: seed to initialize the neural network
        init_sample: samples used to initialize the network
        """
        super().__init__()

        # everything here except self.batch_size, self.sampler and self.state should remain the same after the instance is created, otherwise it will lead to a bug
        # consider creating a new instance to change anything

        self.sampler = sampler
        assert isinstance(self.sampler, AbstractSampler)

        # initialize the network
        if init_sample is None:
            init_sample = jnp.empty((1, *system_shape))
        self.state = net.init(jax.random.PRNGKey(init_seed), init_sample)  # neural network state
        flattened_params, param_unravel_func = flatten_util.ravel_pytree(self.state['params'])
        self.param_is_complex = jnp.iscomplexobj(
            flattened_params[0])  # this may be improved to keep track of each parameter. Will keep as is for now
        assert not self.param_is_complex, "parameters must be real valued"

        # use the network and all the frozen state to create pure helper functions with var_statePure (immutable)
        self.pure_funcs = SimpleVarStateRealPure(
            net=net, system_shape=system_shape, param_unravel_func=param_unravel_func)

        self.sampler.set_var_state(self)  # pass self to sampler so sampler knows how to sample

    # we can access them, but we cannot set them
    @property
    def system_shape(self):
        return self.pure_funcs.system_shape

    @property
    def nb_sites(self):
        return prod(self.system_shape)

    @property
    def net(self):
        return self.pure_funcs.net

    def update_parameters(self, d_params):
        """
        update the trainable parameters
        """
        if isinstance(d_params, jnp.ndarray):
            # flattend
            d_params = self.pure_funcs.unflatten_parameters(d_params)
        new_params = tree_util.tree_map(jnp.add, self.state['params'], d_params)
        self.state = flax.core.copy(self.state, add_or_replace={'params': new_params})

    def sample(self):
        """
        sample from normalized |psi|^2
        """
        return self.sampler()  # same as self.sampler.sample()

    def get_parameters(self, flatten=False):
        """
        get the trainable parameters
        Note we are keeping complex parameters as is
        """
        params = self.state['params']
        if not flatten:
            return params
        else:
            return self.pure_funcs.flatten_parameters(params)

    def set_parameters(self, new_params):
        """
        set the trainable parameters
        Note we are keeping complex parameters as is
        """
        if isinstance(new_params, jnp.ndarray):
            # flattend
            new_params = self.pure_funcs.unflatten_parameters(new_params)
        self.state = flax.core.copy(self.state, add_or_replace={'params': new_params})

    def count_parameters(self):
        """
        count trainable parameters (each complex parameter counted as two parameters)
        """
        return len(self.get_parameters(flatten=True))

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(frozen_dict.unfreeze(self.get_state()), f)

    def load_state(self, path, allow_missing=False):
        if allow_missing:
            current_state = self.get_state()
            with open(path, "rb") as f:
                new_state = pickle.load(f)
            for k, v in current_state.items():
                for k2, v2 in v.items():
                    if k2 not in new_state[k]:
                        warnings.warn(f"key {k}.{k2} not found in the loaded state. Keeping the current value")
                        new_state[k][k2] = v2
            self.set_state(new_state)
        else:
            with open(path, "rb") as f:
                self.set_state(pickle.load(f))
