from abc import ABC, abstractmethod
from typing import (
    Any,
    Tuple,
    Union,
)

import jax.numpy as jnp
import numpy as np

# from jax.experimental import sparse as jexp_sparse

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any


class AbstractPOperator(ABC):
    frozen = False

    def __init__(self):
        """
        init_function freezes the class, so when called using super from subclass, it should be called last.
        """
        self.frozen = True

    def __setattr__(self, name, value):
        """
        try to make this class frozon
        """
        if self.frozen and hasattr(self, name):
            raise AttributeError(f"Cannot modify attribute {name} of a frozen instance")
        super().__setattr__(name, value)

    def __call__(self, var_state: Any, samples: Array, log_psi: Array = None, compile: bool = True) -> Array:
        """
        convinient wrapper of local_energy
        compile: whether to use the compiled version or not
        """
        return self.local_operator(var_state, samples, log_psi, compile)

    @abstractmethod
    def local_operator(self, var_state: Any, samples: Array, log_psi: Array = None, compile: bool = True) -> Array:
        """
        computes the local operator O p / p
        """
        pass
