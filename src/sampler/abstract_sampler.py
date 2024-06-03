from abc import ABC, abstractmethod


class AbstractSampler(ABC):
    is_exact_sampler = False

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def thermalize(self, *args, **kwargs):
        """
        just to be compatible with code designed for mcmc sampler
        """
        return self.sample()

    @abstractmethod
    def set_var_state(self, var_state):
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
