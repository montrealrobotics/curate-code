from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


PLUNDER_EPISODE_STEPS = 4000
PLUNDER_EASY_PARAMS_P1 = list(range(1, 20+1))
PLUNDER_EASY_PARAMS_P2 = list(range(1, 10+1))
PLUNDER_HARD_PARAMS_P1 = list(range(1, 20+1))
PLUNDER_HARD_PARAMS_P2 = list(range(1, 10+1))
PLUNDER_HARD_PARAMS_P3 = list(range(3+1))
PLUNDER_EASIER_PARAMS_P1 = list(range(1, 8+1))
PLUNDER_EASIER_PARAMS_P2 = list(range(1, 10+1))


class ProcgenEnvPlunderEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='plunder',
            distribution_mode='easy',
            param_values=[[-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Plunder-Easy-v0',
    entry_point=module_path + ':ProcgenEnvPlunderEasy',
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenEnvPlunderEasyTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='plunder',
            distribution_mode='easy',
            param_values=[[-1], [-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Plunder-Easy-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvPlunderEasyTerminal',
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenEnvPlunderHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='plunder',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Plunder-Hard-v0',
    entry_point=module_path + ':ProcgenEnvPlunderHard',
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenEnvPlunderHardTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='plunder',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Plunder-Hard-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvPlunderHardTerminal',
    max_episode_steps=PLUNDER_EPISODE_STEPS,
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
    "easy": PLUNDER_EASY_PARAMS_P1,
    "hard": PLUNDER_HARD_PARAMS_P1,
}

p2_params = {
    "easy": PLUNDER_EASY_PARAMS_P2,
    "hard": PLUNDER_HARD_PARAMS_P2,
}

p3_params = {
    "hard": PLUNDER_HARD_PARAMS_P3,
}

## Easy distribution
mode = distribution_modes_to_use[0]
assert mode == "easy"
p1_params_this_mode = p1_params[mode]
p2_params_this_mode = p2_params[mode]
for rew in reward_modes_to_use:
    for act in action_spaces_to_use:
        for t in p1_params_this_mode:
            for j in p2_params_this_mode:

                mode_str = mode[0].upper() + mode[1:]
                rew_str = ""
                if rew == "terminal":
                    rew_str = "Terminal"
                elif rew != "original":
                    raise NotImplementedError
                if act != "original":
                    raise NotImplementedError

                class_name = f"ProcgenEnvPlunder{mode_str}T{t}to{t}J{j}to{j}{rew_str}"

                class_definition_str = ""
                class_definition_str += (
                    f"class {class_name}(ProcgenParamsEnv):\n"
                    f"    def __init__(self, seed=None, **kwargs):\n"
                    f"        super().__init__(\n"
                    f"            game='plunder',\n"
                    f"            distribution_mode='{mode}',\n"
                    f"            param_values=[[{t}], [{j}]],\n"
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
                gym_id += f"Procgen-Plunder-{mode_str}-T{t}to{t}-J{j}to{j}-"
                if rew == "terminal":
                    gym_id += "Terminal-"
                gym_id += "v0"

                class_registration_str = (
                    f"gym_register(\n"
                    f"    id='{gym_id}',\n"
                    f"    entry_point=module_path + ':{class_name}',\n"
                    f"    max_episode_steps={PLUNDER_EPISODE_STEPS},\n"
                    f")\n"
                    f"\n"
                )
                # print(class_registration_str)
                eval(class_registration_str)

## Hard distribution
mode = distribution_modes_to_use[1]
assert mode == "hard"
p1_params_this_mode = p1_params[mode]
p2_params_this_mode = p2_params[mode]
p3_params_this_mode = p3_params[mode]
for rew in reward_modes_to_use:
    for act in action_spaces_to_use:
        for t in p1_params_this_mode:
            for j in p2_params_this_mode:
                for p in p3_params_this_mode:

                    mode_str = mode[0].upper() + mode[1:]
                    rew_str = ""
                    if rew == "terminal":
                        rew_str = "Terminal"
                    elif rew != "original":
                        raise NotImplementedError
                    if act != "original":
                        raise NotImplementedError

                    class_name = f"ProcgenEnvPlunder{mode_str}T{t}to{t}J{j}to{j}P{p}to{p}{rew_str}"

                    class_definition_str = ""
                    class_definition_str += (
                        f"class {class_name}(ProcgenParamsEnv):\n"
                        f"    def __init__(self, seed=None, **kwargs):\n"
                        f"        super().__init__(\n"
                        f"            game='plunder',\n"
                        f"            distribution_mode='{mode}',\n"
                        f"            param_values=[[{t}], [{j}], [{p}]],\n"
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
                    gym_id += f"Procgen-Plunder-{mode_str}-T{t}to{t}-J{j}to{j}-P{p}to{p}-"
                    if rew == "terminal":
                        gym_id += "Terminal-"
                    gym_id += "v0"

                    class_registration_str = (
                        f"gym_register(\n"
                        f"    id='{gym_id}',\n"
                        f"    entry_point=module_path + ':{class_name}',\n"
                        f"    max_episode_steps={PLUNDER_EPISODE_STEPS},\n"
                        f")\n"
                        f"\n"
                    )
                    # print(class_registration_str)
                    eval(class_registration_str)
