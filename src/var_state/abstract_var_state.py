import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Tuple,
    Union,
)

import jax

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


@dataclass(frozen=True)  # to make it immutable and hashable
class AbstractVarStatePure(ABC):

    @abstractmethod
    def evaluate(self, state: Any, samples: Union[Array, Any]) -> Array:
        pass

    @partial(jax.jit, static_argnums=0)
    def evaluate_jitted(self, state: Any, samples: Union[Array, Any]) -> Array:
        return self.evaluate(state, samples)

    @partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=0)
    def evaluate_pmapped(self, state: Any, samples: Union[Array, Any]) -> Array:
        return self.evaluate(state, samples)

    # cannot jit this because it returns functions. should deal with parallel computation internally
    @abstractmethod
    def jvp_vjp_func(self, state: Any, samples: Union[Array, Any]) -> Tuple[Callable, Callable, Array]:
        """
        returns: jvp: callable
                 vjp: callable
                 value
        """
        pass

    @abstractmethod
    def jacobian(self, state: Any, samples: Union[Array, Any]) -> Array:
        """
        return a complex valued matrix or array,
        with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
        """
        pass

    @partial(jax.jit, static_argnums=0)
    def jacobian_jitted(self, state: Any, samples: Union[Array, Any]) -> Array:
        return self.jacobian(state, samples)

    @partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=0)
    def jacobian_pmapped(self, state: Any, samples: Union[Array, Any]) -> Array:
        return self.jacobian(state, samples)

    def log_psi(self, state: Any, samples: Union[Array, Any]) -> Array:
        # assert False
        warnings.warn("log_psi is deprecated. Use evaluate instead.")
        return self.evaluate(state, samples)

    def log_psi_jitted(self, state: Any, samples: Union[Array, Any]) -> Array:
        warnings.warn("log_psi_jitted is deprecated. Use evaluate_jitted instead.")
        return self.evaluate_jitted(state, samples)

    def log_psi_pmapped(self, state: Any, samples: Union[Array, Any]) -> Array:
        warnings.warn("log_psi_pmapped is deprecated. Use evaluate_pmapped instead.")
        return self.evaluate_pmapped(state, samples)

    def jvp_vjp_log_psi_func(self, state: Any, samples: Union[Array, Any]) -> Tuple[Callable, Callable, Array]:
        warnings.warn("jvp_vjp_log_psi_func is deprecated. Use jvp_vjp_func instead.")
        return self.jvp_vjp_func(state, samples)

    def jac_log_psi(self, state: Any, samples: Union[Array, Any]) -> Array:
        warnings.warn("jac_log_psi is deprecated. Use jacobian instead.")
        return self.jacobian(state, samples)

    def jac_log_psi_jitted(self, state: Any, samples: Union[Array, Any]) -> Array:
        warnings.warn("jac_log_psi_jitted is deprecated. Use jacobian_jitted instead.")
        return self.jacobian_jitted(state, samples)

    def jac_log_psi_pmapped(self, state: Any, samples: Union[Array, Any]) -> Array:
        warnings.warn("jac_log_psi_pmapped is deprecated. Use jacobian_pmapped instead.")
        return self.jacobian_pmapped(state, samples)


class AbstractVarState(ABC):

    pure_funcs = None

    def __call__(self, samples):
        return self.evaluate(samples)

    def evaluate(self, samples):
        """
        evaluate the variational state at the provided samples. Can be unnormalized
        """
        return self.pure_funcs.evaluate_pmapped(self.get_state(), samples)

    def jacobian(self, samples):
        """
        returns the jacobian matrix regarding the current samples
        return a complex valued matrix or array,
        with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
        """
        return self.pure_funcs.jacobian_pmapped(self.get_state(), samples)

    def jvp_vjp_func(self, samples, return_value=True, jit=True):
        jvp, vjp, value = self.pure_funcs.jvp_vjp_func(self.get_state(), samples)
        if jit:
            jvp = jax.jit(jvp)
            vjp = jax.jit(vjp)
        if return_value:
            return jvp, vjp, value
        else:
            return jvp, vjp

    def log_psi(self, samples):
        """
        evaluates the log_psi at the provided samples. Can be unnormalized
        """
        warnings.warn('deprecated, use evaluate instead')
        return self.evaluate(samples)

    def jac_log_psi(self, samples):
        """
        returns the jacobian matrix regarding the current samples
        return a complex valued matrix or array,
        with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
        """
        warnings.warn('deprecated, use jacobian instead')
        return self.jacobian(samples)

    def jvp_vjp_log_psi_func(self, samples, return_value=True, jit=True):
        warnings.warn('deprecated, use jvp_vjp_func instead')
        return self.jvp_vjp_func(samples, return_value, jit)

    @abstractmethod
    def sample(self) -> Tuple[Array, Array, Array]:
        """
        sample from normalized |psi|^2
        returns: samples
                 log_psi
                 sqrt_weights
        """
        pass

    @abstractmethod
    def get_parameters(self, flatten=False):
        """
        get the trainable parameters
        """
        pass

    @abstractmethod
    def set_parameters(self, new_params):
        """
        set the trainable parameters
        """
        pass

    @abstractmethod
    def update_parameters(self, d_params):
        """
        update the trainable parameters
        """
        pass

    @abstractmethod
    def count_parameters(self):
        """
        count trainable parameters
        """
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def save_state(self, path):
        pass

    @abstractmethod
    def load_state(self, path):
        pass
