import numpy as np

from .vec_env import VecEnv
from .parallel_wrappers import _flatten_obs, _flatten_list


def get_env_attr(env, attr):
    if hasattr(env, attr):
        return getattr(env, attr)

    while hasattr(env, 'env'):
        env = env.env
        if hasattr(env, attr):
            return getattr(env, attr)

    return None


class LinkedVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel, except there is one "master" environment which has the
    state of the environment that is vectorized, and the rest of the environments as just views for that environment.
    Based off of SubprocVecEnv.
    """

    def __init__(self, env_fn, spaces=None, is_eval=False):
        """
        Arguments:
        env_fn: callable -  function that creates the vectorized environment
        """
        self.waiting = False
        self.closed = False

        self.locals = env_fn()
        self.num_envs = len(self.locals)
        observation_space = self.locals[0].observation_space
        action_space = self.locals[0].action_space
        self.spec = self.locals[0].spec

        self.viewer = None
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

        # Get processed action dim
        self.is_eval = is_eval
        self.processed_action_dim = 1
        if not is_eval:
            self.processed_action_dim = self.locals[0].processed_action_dim

    def step_async(self, action):
        self._assert_not_closed()
        action = np.array_split(action, self.num_envs)
        for local, action in zip(self.locals, action):
            local.step_vec_set_action(action[0])
        self.locals[0].step_vec_all()

        # Reset envs without random resets
        for idx, done in enumerate(self.locals[0].done_from_procgen_after_step):
            if done:
                obs = self.locals[idx].reset()
                self.locals[0].obs_from_procgen_after_step[idx] = obs

        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        obs, rews, dones, infos = self.locals[0].step_vec_get_results()
        self.waiting = False
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        obs = [[local.reset()] for local in self.locals]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        assert not self.waiting
        self.locals[0].env.close()

    def get_complexity_info(self):
        self._assert_not_closed()
        info = [[local.get_complexity_info()] for local in self.locals]
        info = _flatten_list(info)
        return info

    def get_images(self):
        raise NotImplementedError
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [remote.recv() for remote in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def render_to_screen(self):
        raise NotImplementedError
        self._assert_not_closed()
        self.remotes[0].send(('render_to_screen', None))
        return self.remotes[0].recv()

    def max_episode_steps(self):
        raise NotImplementedError
        self._assert_not_closed()
        self.remotes[0].send(('max_episode_steps', None))
        return self.remotes[0].recv()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a LinkedVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

class ParallelAdversarialLinkedVecEnv(LinkedVecEnv):
    def __init__(self, env_fn, adversary=True, is_eval=False):
        super().__init__(env_fn, is_eval=is_eval)
        action_space = self.action_space
        if action_space.__class__.__name__ in ['Box', 'MultiDiscrete']:
            self.action_dim = action_space.shape[0]
        else:
            self.action_dim = 1

        self.adv_action_dim = 0
        if adversary:
            adv_action_space = self.adversary_action_space
            if adv_action_space.__class__.__name__ in ['Box', 'MultiDiscrete']:
                self.adv_action_dim = adv_action_space.shape[0]
            else:
                self.adv_action_dim = 1

    def _should_expand_action(self, action, adversary=False):
        if not adversary:
            action_dim = self.action_dim
        else:
            action_dim = self.adv_action_dim
        # print('expanding actions?', action_dim>1, flush=True)
        return action_dim > 1 or self.processed_action_dim > 1

    def seed_async(self, seed, index):
        raise NotImplementedError
        self._assert_not_closed()
        self.remotes[index].send(('seed', seed))
        self.waiting = True

    def seed_wait(self, index):
        raise NotImplementedError
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        raise NotImplementedError
        self.seed_async(seed, index)
        return self.seed_wait(index)

    def level_seed_async(self, index):
        raise NotImplementedError
        self._assert_not_closed()
        self.remotes[index].send(('level_seed', None))
        self.waiting = True

    def level_seed_wait(self, index):
        raise NotImplementedError
        self._assert_not_closed()
        level_seed = self.remotes[index].recv()
        self.waiting = False
        return level_seed

    def level_seed(self, index):
        raise NotImplementedError
        self.level_seed_async(index)
        return self.level_seed_wait(index)

    # step_adversary
    def step_adversary(self, action):
        if self._should_expand_action(action, adversary=True):
            action = np.expand_dims(action, 1)
        self.step_adversary_async(action)
        return self.step_wait()

    def step_adversary_async(self, action):
        self._assert_not_closed()
        for local, a in zip(self.locals, action):
            local.step_vec_set_action(a[0])
        self.locals[0].step_vec_adversary_all()
        self.waiting = True

    def step_env_async(self, action):
        self._assert_not_closed()
        if self._should_expand_action(action):
            action = np.expand_dims(action, 1)
        for local, a in zip(self.locals, action):
            local.step_vec_set_action(a[0])
        self.locals[0].step_vec_all()

        # Reset agent in env without random resets
        for idx, done in enumerate(self.locals[0].done_from_procgen_after_step):
            if done:
                obs = self.locals[idx].reset_agent()
                self.locals[0].obs_from_procgen_after_step[idx] = obs

        self.waiting = True

    def step_env_reset_random_async(self, action):
        raise NotImplementedError
        self._assert_not_closed()
        if self._should_expand_action(action):
            action = np.expand_dims(action, 1)
        [remote.send(('step_env_reset_random', a)) for remote, a in zip(self.remotes, action)]
        self.waiting = True

    # reset_agent
    def reset_agent(self):
        self._assert_not_closed()
        self.waiting = True
        obs = [[local.reset_agent()] for local in self.locals]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_random
    def reset_random(self):
        self._assert_not_closed()
        self.waiting = True
        obs = [[local.reset_random()] for local in self.locals]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_to_level
    def reset_to_level(self, level, index):
        self._assert_not_closed()
        self.waiting = True
        obs = [self.locals[index].reset_to_level(level)]
        self.waiting = False
        return _flatten_obs(obs)

    def reset_to_level_batch(self, level):
        self._assert_not_closed()
        self.waiting = True
        obs = [[local.reset_to_level(level[i])] for i, local in enumerate(self.locals)]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_to_param
    def reset_to_params(self, params, index):
        self._assert_not_closed()
        self.waiting = True
        obs = [self.locals[index].reset_to_params(params)]
        self.waiting = False
        return _flatten_obs(obs)

    def reset_to_params_batch(self, params_batch):
        self._assert_not_closed()
        self.waiting = True
        obs = [[local.reset_to_params(params_batch[i])] for i, local in enumerate(self.locals)]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # mutate level
    def mutate_level(self, num_edits):
        self._assert_not_closed()
        self.waiting = True
        obs = [[local.mutate_level(num_edits)] for i, local in enumerate(self.locals)]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # observation_space
    def get_observation_space(self):
        self._assert_not_closed()
        obs_space = self.locals[0].observation_space
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    # adversary_observation_space
    def get_adversary_observation_space(self):
        self._assert_not_closed()
        obs_space = self.locals[0].adversary_observation_space
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    def get_adversary_action_space(self):
        self._assert_not_closed()
        action_dim = self.locals[0].adversary_action_space
        return action_dim

    def sample_adversary_action_space(self, index):
        raise NotImplementedError
        self._assert_not_closed()
        self.remotes[index].send(('sample_adversary_action_space', None))
        self.waiting = True
        sampled_adversary_action = self.remotes[index].recv()[0]
        self.waiting = False
        return sampled_adversary_action

    def get_max_episode_steps(self):
        self._assert_not_closed()
        self.waiting = True
        max_episode_steps = get_env_attr(self.locals[0], "_max_episode_steps")
        self.waiting = False
        return max_episode_steps

    # Generic getter
    def local_attr(self, name, data=None, flatten=False, index=None):
        self._assert_not_closed()

        if index is None or len(index) == 0:
            locals = self.locals
        else:
            locals = [self.locals[i] for i in index]

        self.waiting = True
        result = None

        attrs = [getattr(local, name) for local in locals]
        is_callable = hasattr(attrs[0], '__call__')
        if is_callable:
            if not hasattr(data, '__len__'):
                data = [data]*len(attrs)
            result = [[attr(d)] if d is not None else [attr()] for attr, d in zip(attrs, data)]
        else:
            result = [[attr] for attr in attrs]

        self.waiting = False
        return _flatten_list(result) if flatten else result

    def get_seed(self):
        return self.local_attr('seed_value', flatten=True)

    def set_seed(self, seeds):
        return self.local_attr('seed', data=seeds, flatten=True)

    def get_level(self):
        raise NotImplementedError
        levels = self.remote_attr('level')
        return [l[0] for l in levels]  # flatten

    def get_encodings(self, index=None):
        return self.local_attr('encoding', flatten=True, index=index)

    # Navigation-specific
    def get_distance_to_goal(self):
        return self.local_attr('distance_to_goal', flatten=True)

    def get_passable(self):
        return self.local_attr('passable', flatten=True)

    def get_shortest_path_length(self):
        return self.local_attr('shortest_path_length', flatten=True)

    # ALP-GMM-specific
    def reset_alp_gmm(self, levels):
        raise NotImplementedError
        self._assert_not_closed()
        [remote.send(('reset_alp_gmm', levels[i])) for i, remote in enumerate(self.remotes)]
        self.waiting = True
        self._assert_not_closed()
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # === Multigrid-specific ===
    def get_num_blocks(self):
        raise NotImplementedError
        return self.remote_attr('n_clutter_placed', flatten=True)

    def __getattr__(self, name):
        if name == 'observation_space':
            return self.get_observation_space()
        elif name == 'adversary_observation_space':
            return self.get_adversary_observation_space()
        elif name == 'adversary_action_space':
            return self.get_adversary_action_space()
        elif name == 'max_steps':
            return self.get_max_steps()
        else:
            raise NotImplementedError
            return self.__getattribute__(name)
