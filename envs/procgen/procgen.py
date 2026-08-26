import numpy as np

from gym import spaces

from envs.registration import register as gym_register

from procgen import ProcgenEnv


class ProcgenParamsEnv(ProcgenEnv):

    def __init__(
            self,
            game,
            param_values,
            num_procs=1,
            distribution_mode="easy",
            terminal_reward_mode=False,
            seed=None,
            resource_root=None,
            prebuilt_root=None,
            level_options_mode=0,
        ):
        self.num_procs = num_procs
        self.is_vectorized = True if self.num_procs > 1 else False
        num_threads = 4 if self.is_vectorized else 0

        super().__init__(
            num_procs,
            game,
            distribution_mode=distribution_mode,
            terminal_reward_mode=terminal_reward_mode,
            rand_seed=seed,
            num_threads=num_threads,
            resource_root=resource_root,
            prebuilt_root=prebuilt_root,
            level_options_mode=level_options_mode,
        )

        self.seed(seed)

        self.num_params = len(param_values)
        for p in param_values:
            assert not np.isscalar(p), \
                f"Expected param_values to be a list of lists."
        self.param_values = param_values
        self.num_params = len(self.param_values)

        # Redefine observation space
        assert len(self.observation_space.spaces) == 1 and 'rgb' in self.observation_space.spaces
        obs_spaces = {
            'image': self.observation_space['rgb'],
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # We assume all levels are passable
        self.passable = [True for _ in range(self.num_procs)]

        # We now call reset to get the new level seed and sampled options
        self.base_level_seeds = [None for _ in range(self.num_procs)]
        self.base_level_params = [None for _ in range(self.num_procs)]
        self.is_first_step = [None for _ in range(self.num_procs)]
        self.level_encodings = [None for _ in range(self.num_procs)]
        self.step_actions = [None for _ in range(self.num_procs)]
        self.obs_from_procgen_after_step = [None for _ in range(self.num_procs)]
        self.reward_from_procgen_after_step = [None for _ in range(self.num_procs)]
        self.done_from_procgen_after_step = [None for _ in range(self.num_procs)]
        self.info_from_procgen_after_step = [None for _ in range(self.num_procs)]
        for idx in range(self.num_procs):
            self.reset(env_idx=idx)

    @staticmethod
    def _convert_to_image_obs(obs_from_procgen, env_idx=0):
        image_obs = obs_from_procgen['rgb'][env_idx]
        obs = {
            'image': image_obs,
        }
        return obs

    def encoding(self, env_idx=0):
        assert self.is_first_step[env_idx], \
            "Environment must be on first step for encoding to be valid."

        # level encoding is fully specified by the level and level options
        assert self.base_level_seeds[env_idx] is not None
        assert self.base_level_params[env_idx] is not None

        encoded_level = np.array(np.r_[self.base_level_seeds[env_idx], self.base_level_params[env_idx]], dtype=np.int64)
        return encoded_level

    def get_complexity_info(self, env_idx=0):
        complexity_info = {}
        for p in range(self.num_params):
            complexity_info[f"level_option_{p+1}"] = self.level_encodings[env_idx][p+1]

        return complexity_info

    def seed(self, seed, env_idx=0):
        if env_idx == 0:
            self.rng = np.random.default_rng(seed=seed)
            self.seed_value = seed
        else:
            print(f'Warning: seed only works for env_idx == 0 currently, but env_idx == {env_idx} was called.')
        return [self.seed_value]

    def reset(self, base_level_seed=None, params=None, env_idx=0):
        # set level seed
        sampled_level_seed = self.rng.integers(2147483647 + 1)
        if base_level_seed is None:
            self.base_level_seeds[env_idx] = sampled_level_seed
        else:
            self.base_level_seeds[env_idx] = base_level_seed
        self.reset_start_level(self.base_level_seeds[env_idx], env_idx=env_idx)

        # set level options
        sampled_params = [self.rng.choice(v) for v in self.param_values]
        if params is None:
            self.base_level_params[env_idx] = sampled_params
        else:
            assert len(params) == self.num_params
            self.base_level_params[env_idx] = params
        level_params_for_update_level_options = [self.base_level_params[env_idx][p] if p < self.num_params else -1 for p in range(4)]
        self.update_level_options(*level_params_for_update_level_options, env_idx=env_idx)
        obs = self.gen_obs(env_idx=env_idx)
        self.is_first_step[env_idx] = True
        self.level_encodings[env_idx] = self.encoding(env_idx=env_idx)
        return obs

    def gen_obs(self, env_idx=0):
        obs_from_procgen = super().observe()
        obs = self._convert_to_image_obs(obs_from_procgen, env_idx=env_idx)
        return obs

    def step(self, action):
        assert not self.is_vectorized
        action_for_procgen = np.array(action).reshape(-1)    # reshape(-1) ensures the array is flattened
        obs_from_procgen, reward_from_procgen, done_from_procgen, info_from_procgen = super().step(np.array(action_for_procgen))

        obs = self._convert_to_image_obs(obs_from_procgen, env_idx=0)
        reward = reward_from_procgen[0]
        done = done_from_procgen[0]
        info = info_from_procgen[0]

        self.is_first_step[0] = False

        return obs, reward, done, info

    def step_vec_set_action(self, action, env_idx=0):
        assert self.is_vectorized
        self.step_actions[env_idx] = action

    def step_vec_all(self, env_idx=0):
        assert self.is_vectorized
        assert env_idx == 0

        for a in self.step_actions:
            assert a is not None

        action_for_procgen = np.array(self.step_actions).reshape(-1)    # reshape(-1) ensures the array is flattened
        obs_from_procgen, reward_from_procgen, done_from_procgen, info_from_procgen = super().step(action_for_procgen)
        for idx in range(self.num_procs):
            obs = self._convert_to_image_obs(obs_from_procgen, env_idx=idx)
            self.obs_from_procgen_after_step[idx] = obs
            self.reward_from_procgen_after_step[idx] = reward_from_procgen[idx]
            self.done_from_procgen_after_step[idx] = done_from_procgen[idx]
            self.info_from_procgen_after_step[idx] = info_from_procgen[idx]

        for idx in range(self.num_procs):
            self.step_actions[idx] = None

    def step_vec_get_results(self, env_idx=0):
        assert self.is_vectorized
        assert env_idx == 0

        obs = tuple(self.obs_from_procgen_after_step)
        for idx in range(self.num_procs):
            self.obs_from_procgen_after_step[idx] = None

        rewards = tuple(self.reward_from_procgen_after_step)
        for idx in range(self.num_procs):
            self.reward_from_procgen_after_step[idx] = None

        dones = tuple(self.done_from_procgen_after_step)
        for idx in range(self.num_procs):
            self.done_from_procgen_after_step[idx] = None

        infos = tuple(self.info_from_procgen_after_step)
        for idx in range(self.num_procs):
            self.info_from_procgen_after_step[idx] = None

        return obs, rewards, dones, infos
