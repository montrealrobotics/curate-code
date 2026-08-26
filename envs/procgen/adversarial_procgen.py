import numpy as np
from copy import deepcopy

from gym import spaces

from .procgen import ProcgenParamsEnv


class AdversarialProcgenParamsEnv(ProcgenParamsEnv):

    def __init__(
            self,
            game,
            param_values,
            num_procs=1,
            distribution_mode="easy",
            terminal_reward_mode=False,
            min_param_mode=False,
            seed=None,
            random_z_dim=50,
            resource_root=None,
            prebuilt_root=None,
            level_options_mode=0,
    ):
        self.random_z_dim = random_z_dim
        self.min_param_mode = min_param_mode

        self.adversary_step_counts = [None for _ in range(num_procs)]

        super().__init__(
            game,
            param_values,
            num_procs=num_procs,
            distribution_mode=distribution_mode,
            terminal_reward_mode=terminal_reward_mode,
            seed=seed,
            resource_root=resource_root,
            prebuilt_root=prebuilt_root,
            level_options_mode=level_options_mode,
        )

        # from multigrid adversarial
        self.adversary_max_steps = 1
        self.adversary_action_space = spaces.MultiDiscrete([len(p) for p in self.param_values])
        self.adversary_action_space_is_unidimensional = self.adversary_action_space.shape[0] == 1

        self.adversary_image_obs_space = self.observation_space['image']
        self.adversary_ts_obs_space = spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8',
        )
        self.adversary_randomz_obs_space = spaces.Box(
            low=0, high=1.0, shape=(self.random_z_dim,), dtype=np.float32,
        )
        self.adversary_observation_space = spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space
        })

    @property
    def processed_action_dim(self):
        assert self.action_space.__class__.__name__ == 'Discrete'
        return 1

    """ based on multigrid adversarial """
    def _generate_random_z(self):
        random_z = self.rng.uniform(size=(self.random_z_dim,)).astype(np.float32)
        return random_z

    def reset(self, env_idx=0):
        # Reset to an easiest task
        min_params = [p[0] for p in self.param_values]
        self.reset_to_params(params=min_params, env_idx=env_idx)
        self.adversary_step_counts[env_idx] = 0
        adv_obs = self.gen_adversary_obs(env_idx=env_idx)
        return adv_obs

    def reset_agent(self, env_idx=0):
        return self.restart_level(env_idx=env_idx)

    def restart_level(self, env_idx=0):
        base_level_seed = self.level_encodings[env_idx][0]
        level_options = self.level_encodings[env_idx][1:]
        obs = super().reset(base_level_seed=base_level_seed, params=level_options, env_idx=env_idx)
        return obs

    def reset_to_level(self, level, env_idx=0):
        obs = self.reset_to_encoding(level, env_idx=env_idx)
        return obs

    def reset_to_encoding(self, encoded_level, env_idx=0):
        base_level_seed = encoded_level[0]
        level_params = encoded_level[1:]

        super().reset(base_level_seed=base_level_seed, params=level_params, env_idx=env_idx)
        return self.reset_agent()

    def mutate_level(self, num_edits=1, env_idx=0):
        assert self.is_first_step, \
            f"Mutating a level is only possible before the agent has acted."

        base_level_seed = self.level_encodings[env_idx][0]
        mutated_params = deepcopy(self.level_encodings[env_idx][1:])

        for _ in range(num_edits):
            edits = []
            for p in range(self.num_params):

                if self.param_values[p][0] < mutated_params[p]:
                    # can decrease parameter
                    edits.append((p, -1))
                if mutated_params[p] < self.param_values[p][-1]:
                    # can increase parameter
                    edits.append((p, 1))

            x_edit = self.rng.integers(len(edits))
            selected_edit = edits[x_edit]
            selected_edit_param, selected_edit_delta = selected_edit

            idx_curr = self.param_values[selected_edit_param].index(mutated_params[selected_edit_param])
            mutated_params[selected_edit_param] = self.param_values[selected_edit_param][idx_curr + selected_edit_delta]

        super().reset(base_level_seed=base_level_seed, params=mutated_params, env_idx=env_idx)
        return self.reset_agent()

    def step_adversary(self, action, env_idx=0):
        if not self.min_param_mode:
            adv_action = [action] if self.adversary_action_space_is_unidimensional else action
            adversary_params = [self.param_values[p][idx] for p, idx in enumerate(adv_action)]
            base_level_seed = self.level_encodings[env_idx][0]
            super().reset(base_level_seed=base_level_seed, params=adversary_params, env_idx=env_idx)

        self.adversary_step_counts[env_idx] += 1

        if self.adversary_step_counts[env_idx] == self.adversary_max_steps:
            self.reset_agent(env_idx=env_idx)

        adv_obs = self.gen_adversary_obs(env_idx=env_idx)
        adv_done = self.adversary_step_counts[env_idx] >= self.adversary_max_steps
        adv_reward = 0.
        adv_info = {}

        return adv_obs, adv_reward, adv_done, adv_info

    def step_vec_adversary_all(self, env_idx=0):
        assert self.is_vectorized
        assert env_idx == 0

        for a in self.step_actions:
            assert a is not None

        assert len(self.step_actions) == self.num_procs

        for idx, a in zip(range(self.num_procs), self.step_actions):
            adv_obs, adv_reward, adv_done, adv_info = self.step_adversary(a, env_idx=idx)
            self.obs_from_procgen_after_step[idx] = adv_obs
            self.reward_from_procgen_after_step[idx] = adv_reward
            self.done_from_procgen_after_step[idx] = adv_done
            self.info_from_procgen_after_step[idx] = adv_info

        for idx in range(self.num_procs):
            self.step_actions[idx] = None

    def reset_random(self, params=None, env_idx=0):
        if self.min_param_mode:
            if params is None:
                # Enforce min params
                params = [p[0] for p in self.param_values]
            else:
                # This function should only work if we give it min params
                for idx_p, p in enumerate(params):
                    assert p == self.param_values[idx_p][0]
        obs = super().reset(params=params, env_idx=env_idx)
        return obs

    def reset_to_params(self, params, env_idx=0):
        return self.reset_random(params=params, env_idx=env_idx)

    def gen_adversary_obs(self, env_idx=0):
        level_image = self.gen_obs(env_idx=env_idx)['image']
        adv_obs = {
            'image': level_image,
            'time_step': [self.adversary_step_counts[env_idx]],
            'random_z': self._generate_random_z(),
        }
        return adv_obs
