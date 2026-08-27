from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


CAVEFLYER_EPISODE_STEPS = 1000
CAVEFLYER_EASY_PARAMS_P1 = list(range(3+1))
CAVEFLYER_HARD_PARAMS_P1 = list(range(3+1))


class ProcgenEnvCaveFlyerEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='caveflyer',
            distribution_mode='easy',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CaveFlyer-Easy-v0',
    entry_point=module_path + ':ProcgenEnvCaveFlyerEasy',
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenEnvCaveFlyerEasyTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='caveflyer',
            distribution_mode='easy',
            param_values=[[-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CaveFlyer-Easy-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvCaveFlyerEasyTerminal',
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenEnvCaveFlyerHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='caveflyer',
            distribution_mode='hard',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CaveFlyer-Hard-v0',
    entry_point=module_path + ':ProcgenEnvCaveFlyerHard',
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenEnvCaveFlyerHardTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='caveflyer',
            distribution_mode='hard',
            param_values=[[-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CaveFlyer-Hard-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvCaveFlyerHardTerminal',
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
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
    "easy": CAVEFLYER_EASY_PARAMS_P1,
    "hard": CAVEFLYER_HARD_PARAMS_P1,
}

##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    for rew in reward_modes_to_use:
        for act in action_spaces_to_use:
            for o in p1_params_this_mode:

                mode_str = mode[0].upper() + mode[1:]
                rew_str = ""
                if rew == "terminal":
                    rew_str = "Terminal"
                elif rew != "original":
                    raise NotImplementedError
                if act != "original":
                    raise NotImplementedError

                class_name = f"ProcgenEnvCaveFlyer{mode_str}O{o}to{o}{rew_str}"

                class_definition_str = ""
                class_definition_str += (
                    f"class {class_name}(ProcgenParamsEnv):\n"
                    f"    def __init__(self, seed=None, **kwargs):\n"
                    f"        super().__init__(\n"
                    f"            game='caveflyer',\n"
                    f"            distribution_mode='{mode}',\n"
                    f"            param_values=[[{o}]],\n"
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
                gym_id += f"Procgen-CaveFlyer-{mode_str}-O{o}to{o}-"
                if rew == "terminal":
                    gym_id += "Terminal-"
                gym_id += "v0"

                class_registration_str = (
                    f"gym_register(\n"
                    f"    id='{gym_id}',\n"
                    f"    entry_point=module_path + ':{class_name}',\n"
                    f"    max_episode_steps={CAVEFLYER_EPISODE_STEPS},\n"
                    f")\n"
                    f"\n"
                )
                # print(class_registration_str)
                eval(class_registration_str)
