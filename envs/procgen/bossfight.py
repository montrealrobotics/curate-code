from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


BOSSFIGHT_EPISODE_STEPS = 4000
BOSSFIGHT_EASY_PARAMS_P1 = list(range(1, 9+1))
BOSSFIGHT_EASY_PARAMS_P2 = list(range(1, 5+1))
BOSSFIGHT_EASY_PARAMS_P3 = list(range(2, 3+1))
BOSSFIGHT_HARD_PARAMS_P1 = list(range(1, 9+1))
BOSSFIGHT_HARD_PARAMS_P2 = list(range(1, 5+1))
BOSSFIGHT_HARD_PARAMS_P3 = list(range(2, 5+1))
BOSSFIGHT_EASIER_PARAMS_P1 = list(range(1, 2+1))
BOSSFIGHT_EASIER_PARAMS_P2 = list(range(1, 2+1))
BOSSFIGHT_EASIER_PARAMS_P3 = list(range(2, 3+1))


class ProcgenEnvBossFightEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='bossfight',
            distribution_mode='easy',
            param_values=[[-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-BossFight-Easy-v0',
    entry_point=module_path + ':ProcgenEnvBossFightEasy',
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenEnvBossFightHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='bossfight',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-BossFight-Hard-v0',
    entry_point=module_path + ':ProcgenEnvBossFightHard',
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


# Single-parameter environments

distribution_modes_to_use = [
    "easy",
    "hard",
]

reward_modes_to_use = [
    "original",
    "terminal",
]

action_spaces_to_use = [
    "original",
]

p1_params = {
    "easy": BOSSFIGHT_EASY_PARAMS_P1,
    "hard": BOSSFIGHT_HARD_PARAMS_P1,
}

p2_params = {
    "easy": BOSSFIGHT_EASY_PARAMS_P2,
    "hard": BOSSFIGHT_HARD_PARAMS_P2,
}

p3_params = {
    "easy": BOSSFIGHT_EASY_PARAMS_P3,
    "hard": BOSSFIGHT_HARD_PARAMS_P3,
}


##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    p2_params_this_mode = p2_params[mode]
    p3_params_this_mode = p3_params[mode]
    for rew in reward_modes_to_use:
        for act in action_spaces_to_use:
            for h in p1_params_this_mode:
                for r in p2_params_this_mode:
                    for i in p3_params_this_mode:

                        mode_str = mode[0].upper() + mode[1:]
                        rew_str = ""
                        if rew == "terminal":
                            rew_str = "Terminal"
                        elif rew != "original":
                            raise NotImplementedError
                        if act != "original":
                            raise NotImplementedError

                        class_name = f"ProcgenEnvBossFight{mode_str}H{h}to{h}R{r}to{r}I{i}to{i}{rew_str}"

                        class_definition_str = ""
                        class_definition_str += (
                            f"class {class_name}(ProcgenParamsEnv):\n"
                            f"    def __init__(self, seed=None, **kwargs):\n"
                            f"        super().__init__(\n"
                            f"            game='bossfight',\n"
                            f"            distribution_mode='{mode}',\n"
                            f"            param_values=[[{h}], [{r}], [{i}]],\n"
                        )
                        if rew == "terminal":
                            class_definition_str += \
                                f"            terminal_reward_mode=True,\n"
                        class_definition_str += (
                            f"            seed=seed,\n"
                            f"            **kwargs,\n"
                            f"        )\n"
                            f"\n"
                        )
                        # print(class_definition_str)
                        exec(class_definition_str)

                        gym_id = ""
                        gym_id += f"Procgen-BossFight-{mode_str}-H{h}to{h}-R{r}to{r}-I{i}to{i}-"
                        if rew == "terminal":
                            gym_id += "Terminal-"
                        gym_id += "v0"

                        class_registration_str = (
                            f"gym_register(\n"
                            f"    id='{gym_id}',\n"
                            f"    entry_point=module_path + ':{class_name}',\n"
                            f"    max_episode_steps={BOSSFIGHT_EPISODE_STEPS},\n"
                            f")\n"
                            f"\n"
                        )
                        # print(class_registration_str)
                        eval(class_registration_str)
