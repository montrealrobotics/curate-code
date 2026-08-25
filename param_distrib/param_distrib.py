from abc import ABC, abstractmethod

import importlib
import numpy as np


def load_param_distrib(strategy, env_name, **kwargs):
    module = importlib.import_module('param_distrib')
    param_distrib = None
    try:
        param_distrib_class = getattr(module, strategy)
        param_distrib = param_distrib_class(strategy, env_name, **kwargs)
    except Exception as e:
        print(f"An error occurred when trying to load strategy {strategy}: {e}")
        raise e

    return param_distrib


class ParamDistrib(ABC):
    def __init__(
            self,
            strategy,
            env_name,
            num_processes,
            seed=0,
            enable_optimizer_resets=False,
            **kwargs,
            ):

        self.strategy = strategy
        self.env_name = env_name
        self.update_ctr = 0
        self.num_processes = num_processes
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.optimizer_resets_are_supported = False
        self.enable_optimizer_resets = enable_optimizer_resets
        self.reset_optimizer_trigger = False

    def initialize(self, *args, **kwargs):
        return self._initialize(*args, **kwargs)

    def _initialize(self, *args, **kwargs):
        return True

    def sample(self, index=None):
        return self._sample(index)

    @abstractmethod
    def _sample(self, index=None):
        pass

    def update(self, returns=None, **kwargs):
        # We preincrement the update counter, treating any times when update_ctr == 0 as the initialization period
        self.update_ctr += 1
        return self._update(returns, **kwargs)

    @abstractmethod
    def _update(self, returns=None, **kwargs):
        pass

    def get_stats(self):
        stats = dict(
            update_ctr=self.update_ctr,
        )
        specific_stats = self._get_stats()
        stats.update(specific_stats)
        return stats

    def _get_stats(self):
        return {}

    def ready_for_optimizer_reset(self):
        return self.optimizer_resets_are_supported and self.enable_optimizer_resets and self.reset_optimizer_trigger

    def check_and_reset_optimizer(self, optimizer):
        if self.ready_for_optimizer_reset():
            optimizer.zero_grad()
            optimizer.state.clear()
            self.reset_optimizer_trigger = False


class ConstantParamDistrib(ParamDistrib):

    def __init__(
            self,
            strategy,
            env_name,
            value,
            **kwargs,
        ):
        super().__init__(strategy, env_name, **kwargs)

        self.is_domain_discrete_env_param = True if (
            env_name.startswith('MultiGrid') or
            env_name.startswith('MiniGrid') or
            env_name.startswith('Procgen')
        ) else False

        self.is_domain_scalar_env_param = True if (
            env_name.startswith('MultiGrid') or
            env_name.startswith('MiniGrid')
        ) else False

        value_as_list = [value] if np.isscalar(value) else value

        if self.is_domain_discrete_env_param:
            for v in value_as_list:
                assert isinstance(v, int), \
                    f"For MultiGrid, MiniGrid, and Procgen environments, \"value\" must be an int or list of ints, but parameter {v} is a {type(v)}."

        num_params = len(value_as_list)
        self.value = None
        if self.is_domain_scalar_env_param:
            assert num_params == 1
            self.value = value_as_list[0]
        else:
            self.value = value_as_list

    def _sample(self, index=None):
        return self.value

    def _update(self, returns=None, **kwargs):
        return True

    def _get_stats(self):
        stats = dict(
            value=self.value,
        )
        return stats


