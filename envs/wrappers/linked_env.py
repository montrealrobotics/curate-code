from abc import ABC


class LinkedEnv(ABC):

    def __init__(self, env, env_idx):
        self.env = env
        self.env_idx = env_idx

    def __getattr__(self, name):
        if name == "env":
            return self.env
        elif name == "env_id":
            return self.env_idx
        elif name == "_max_episode_steps":
            # can't use __getattr__ for private attributes with the gym api
            return self.env._max_episode_steps
        else:
            attr = self.env.__getattr__(name)
            if hasattr(attr, "__call__"):

                def specify_env(*args, **kwargs):
                    result = attr(*args, **kwargs, env_idx=self.env_idx)
                    return result

                return specify_env
            else:
                return attr
