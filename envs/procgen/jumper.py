from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


JUMPER_EPISODE_STEPS = 1000
JUMPER_EASY_PARAMS_P1 = list(range(20+1))
JUMPER_HARD_PARAMS_P1 = list(range(20+1))


class ProcgenEnvJumperEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='jumper',
            distribution_mode='easy',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )

gym_register(
    id='Procgen-Jumper-Easy-v0',
    entry_point=module_path + ':ProcgenEnvJumperEasy',
    max_episode_steps=JUMPER_EPISODE_STEPS,
)


class ProcgenEnvJumperHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='jumper',
            distribution_mode='hard',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Jumper-Hard-v0',
    entry_point=module_path + ':ProcgenEnvJumperHard',
    max_episode_steps=JUMPER_EPISODE_STEPS,
)


# Single-parameter environments

distribution_modes_to_use = [
    "easy",
    "hard",
]

action_spaces_to_use = [
    "original",
]

p1_params = {
    "easy": JUMPER_EASY_PARAMS_P1,
    "hard": JUMPER_HARD_PARAMS_P1,
}

##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    for act in action_spaces_to_use:
        for s in p1_params_this_mode:

            mode_str = mode[0].upper() + mode[1:]
            if act != "original":
                raise NotImplementedError

            class_name = f"ProcgenEnvJumper{mode_str}S{s}to{s}"

            class_definition_str = ""
            class_definition_str += (
                f"class {class_name}(ProcgenParamsEnv):\n"
                f"    def __init__(self, seed=None, **kwargs):\n"
                f"        super().__init__(\n"
                f"            game='jumper',\n"
                f"            distribution_mode='{mode}',\n"
                f"            param_values=[[{s}]],\n"
            )
            class_definition_str += (
                f"            seed=seed,\n"
                f"            **kwargs,\n"
                f"        )\n"
                f"\n"
            )
            # print(class_definition_str)
            exec(class_definition_str)

            gym_id = ""
            gym_id += f"Procgen-Jumper-{mode_str}-S{s}to{s}-"
            gym_id += "v0"

            class_registration_str = (
                f"gym_register(\n"
                f"    id='{gym_id}',\n"
                f"    entry_point=module_path + ':{class_name}',\n"
                f"    max_episode_steps={JUMPER_EPISODE_STEPS},\n"
                f")\n"
                f"\n"
            )
            # print(class_registration_str)
            eval(class_registration_str)
