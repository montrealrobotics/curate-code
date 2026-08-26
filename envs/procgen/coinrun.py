from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


COINRUN_EPISODE_STEPS = 1000
COINRUN_EASY_PARAMS_P1 = list(range(1, 3+1))
COINRUN_EASY_PARAMS_P2 = list(range(1, 5+1))
COINRUN_HARD_PARAMS_P1 = list(range(1, 3+1))
COINRUN_HARD_PARAMS_P2 = list(range(1, 5+1))


class ProcgenEnvCoinRunEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='coinrun',
            distribution_mode='easy',
            param_values=[[-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CoinRun-Easy-v0',
    entry_point=module_path + ':ProcgenEnvCoinRunEasy',
    max_episode_steps=COINRUN_EPISODE_STEPS,
)


class ProcgenEnvCoinRunHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='coinrun',
            distribution_mode='hard',
            param_values=[[-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-CoinRun-Hard-v0',
    entry_point=module_path + ':ProcgenEnvCoinRunHard',
    max_episode_steps=COINRUN_EPISODE_STEPS,
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
    "easy": COINRUN_EASY_PARAMS_P1,
    "hard": COINRUN_HARD_PARAMS_P1,
}

p2_params = {
    "easy": COINRUN_EASY_PARAMS_P2,
    "hard": COINRUN_HARD_PARAMS_P2,
}

##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    p2_params_this_mode = p2_params[mode]
    for act in action_spaces_to_use:
        for d in p1_params_this_mode:
            for s in p2_params_this_mode:

                mode_str = mode[0].upper() + mode[1:]
                if act != "original":
                    raise NotImplementedError

                class_name = f"ProcgenEnvCoinRun{mode_str}D{d}to{d}S{s}to{s}"

                class_definition_str = ""
                class_definition_str += (
                    f"class {class_name}(ProcgenParamsEnv):\n"
                    f"    def __init__(self, seed=None, **kwargs):\n"
                    f"        super().__init__(\n"
                    f"            game='coinrun',\n"
                    f"            distribution_mode='{mode}',\n"
                    f"            param_values=[[{d}], [{s}]],\n"
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
                gym_id += f"Procgen-CoinRun-{mode_str}-D{d}to{d}-S{s}to{s}-"
                gym_id += "v0"

                class_registration_str = (
                    f"gym_register(\n"
                    f"    id='{gym_id}',\n"
                    f"    entry_point=module_path + ':{class_name}',\n"
                    f"    max_episode_steps={COINRUN_EPISODE_STEPS},\n"
                    f")\n"
                    f"\n"
                )
                # print(class_registration_str)
                eval(class_registration_str)

