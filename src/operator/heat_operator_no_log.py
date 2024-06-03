from functools import partial
from typing import (
    Any,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np

# from jax.experimental import sparse as jexp_sparse
from .abstract_p_operator import AbstractPOperator

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any


class HeatOperatorNoLog(AbstractPOperator):

    def __init__(self, nb_dims: int, drift_coefs: Array, diffusion_coefs: Array, check_validity=True):
        """
        nb_dims: number of dimensions
        drift_coefs (mu_i): array of shape (nb_dims)
        diffusion_coefs (D_ij): array of shape (nb_dims, nb_dims)
        check_validity: whether to check the validity of the diffusion matrix
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        super().__init__()
        self.nb_dims = nb_dims
        self.drift_coefs = drift_coefs
        self.diffusion_coefs = diffusion_coefs
        if check_validity:
            assert np.allclose(self.diffusion_coefs, self.diffusion_coefs.T), "diffusion matrix must be symmetric"
            assert np.all(np.linalg.eigvals(self.diffusion_coefs) > -1e-7), "diffusion matrix must be positive definite"
            assert self.drift_coefs.shape == (self.nb_dims,), "drift coefs must be of shape (nb_dims,)"
            assert self.diffusion_coefs.shape == (
                self.nb_dims, self.nb_dims), "diffusion coefs must be of shape (nb_dims, nb_dims)"
        # not used for now
        ((self.nonzero_drift_dims, self.nonzero_drift_coefs),
         (self.nonzero_diffusion_dims, self.nonzero_diffusion_coefs)) = self.get_nonzero_coefs(self.drift_coefs,
                                                                                               self.diffusion_coefs)

    def get_nonzero_coefs(self, drift_coefs, diffusion_coefs):
        """
        returns the nonzero coefficients and dims of the drift vector and diffusion matrix
        """
        nonzero_drift_dims = np.where(drift_coefs != 0)[0]
        nonzero_diffusion_dims = np.where(diffusion_coefs != 0)
        nonzero_drift_coefs = drift_coefs[nonzero_drift_dims]
        nonzero_diffusion_coefs = diffusion_coefs[nonzero_diffusion_dims[0], nonzero_diffusion_dims[1]]
        return (nonzero_drift_dims, nonzero_drift_coefs), (
            nonzero_diffusion_dims, nonzero_diffusion_coefs)  # MAYBE CONSIDER TAKING ADVANTAGE OF SYMMETRIC MATRIX

    def local_operator(self, var_state: Any, samples: Array, values: Array = None, compile: bool = True) -> Array:
        """
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        if compile:
            return self.local_operator_compiled(var_state, samples, values)
        else:
            return self.local_operator_uncompiled(var_state, samples, values)

    def local_operator_pure(self, var_state_pure: Any, samples: Any, values: Array, state: PyTreeDef) -> Array:
        """
        a pure function version of computing the local energy.
        will be pmapped and compiled.
        however, compilation unrolls the for loop. unclear how much gain using this function
        we are writing the (possibly) less efficient version for now
        Not that we are computing O u(x)
        """
        u_func = lambda state, samples: var_state_pure.evaluate(state, samples[None, ...]).squeeze(0)
        jac_func = jax.jacrev(u_func, argnums=1)
        hes_func = jax.jacfwd(jac_func, argnums=1)
        jac_func = jax.vmap(jac_func, in_axes=(None, 0), out_axes=0)
        hes_func = jax.vmap(hes_func, in_axes=(None, 0), out_axes=0)
        jac = jac_func(state, samples).reshape(samples.shape[0], samples.shape[-1])  # combine system dimensions
        hes = hes_func(state, samples).reshape(samples.shape[0], samples.shape[-1],
                                               samples.shape[-1])  # combine system dimensions
        drift = (jac * self.drift_coefs).sum(axis=-1)
        diffusion = (hes * self.diffusion_coefs).sum(axis=(-1, -2))
        return jnp.clip(-drift + diffusion, a_min=-1e20, a_max=1e20)  # follow the convention of sign

    @partial(jax.pmap, in_axes=(None, None, 0, 0, None), static_broadcasted_argnums=(0, 1))
    def local_operator_pure_pmapped(self, var_state_pure: Any, samples: Any, values: Array, state: PyTreeDef) -> Array:
        return self.local_operator_pure(var_state_pure, samples, values, state)

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_pure_jitted(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     return self.local_energy_pure_unjitted(var_state_pure, samples, log_psi, state)

    def local_operator_compiled(self, var_state: Any, samples: Any, values: Array = None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        # if values is None:
        #     values = var_state.values(samples) # we don't actually need this for now
        return self.local_operator_pure_pmapped(var_state.pure_funcs, samples, values, var_state.get_state())

    def local_operator_uncompiled(self, var_state: Any, samples: Any, values: Array = None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        # if values is None:
        #     values = var_state.values(samples) # we don't actually need this for now
        return self.local_operator_pure(var_state.pure_funcs, samples, values, var_state.get_state())
