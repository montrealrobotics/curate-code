from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


DODGEBALL_EPISODE_STEPS = 1000
DODGEBALL_EASY_PARAMS_P1 = list(range(3, 6+1))
DODGEBALL_HARD_PARAMS_P1 = list(range(3, 6+1))
DODGEBALL_EASIER_PARAMS_P1 = list(range(3+1))


class ProcgenEnvDodgeballEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='dodgeball',
            distribution_mode='easy',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Dodgeball-Easy-v0',
    entry_point=module_path + ':ProcgenEnvDodgeballEasy',
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenEnvDodgeballEasyTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='dodgeball',
            distribution_mode='easy',
            param_values=[[-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Dodgeball-Easy-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvDodgeballEasyTerminal',
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenEnvDodgeballHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='dodgeball',
            distribution_mode='hard',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Dodgeball-Hard-v0',
    entry_point=module_path + ':ProcgenEnvDodgeballHard',
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenEnvDodgeballHardTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='dodgeball',
            distribution_mode='hard',
            param_values=[[-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Dodgeball-Hard-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvDodgeballHardTerminal',
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
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
    "easy": list(set([*DODGEBALL_EASIER_PARAMS_P1, *DODGEBALL_EASY_PARAMS_P1])),
    "hard": DODGEBALL_HARD_PARAMS_P1,
}

##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    for rew in reward_modes_to_use:
        for act in action_spaces_to_use:
            for e in p1_params_this_mode:

                mode_str = mode[0].upper() + mode[1:]
                rew_str = ""
                if rew == "terminal":
                    rew_str = "Terminal"
                elif rew != "original":
                    raise NotImplementedError
                if act != "original":
                    raise NotImplementedError

                class_name = f"ProcgenEnvDodgeball{mode_str}E{e}to{e}{rew_str}"

                class_definition_str = ""
                class_definition_str += (
                    f"class {class_name}(ProcgenParamsEnv):\n"
                    f"    def __init__(self, seed=None, **kwargs):\n"
                    f"        super().__init__(\n"
                    f"            game='dodgeball',\n"
                    f"            distribution_mode='{mode}',\n"
                    f"            param_values=[[{e}]],\n"
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
                gym_id += f"Procgen-Dodgeball-{mode_str}-E{e}to{e}-"
                if rew == "terminal":
                    gym_id += "Terminal-"
                gym_id += "v0"

                class_registration_str = (
                    f"gym_register(\n"
                    f"    id='{gym_id}',\n"
                    f"    entry_point=module_path + ':{class_name}',\n"
                    f"    max_episode_steps={DODGEBALL_EPISODE_STEPS},\n"
                    f")\n"
                    f"\n"
                )
                # print(class_registration_str)
                eval(class_registration_str)
