from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


FRUITBOT_EPISODE_STEPS = 1000
FRUITBOT_EASY_PARAMS_P1 = list(range(1, 5+1))
FRUITBOT_EASY_PARAMS_P2 = list(range(60+1))
FRUITBOT_EASY_PARAMS_P3 = list(range(10+1))
FRUITBOT_HARD_PARAMS_P1 = list(range(1, 10+1))
FRUITBOT_HARD_PARAMS_P2 = list(range(70+1))
FRUITBOT_HARD_PARAMS_P3 = list(range(10+1))
FRUITBOT_HARD_PARAMS_P4 = list(range(5+1))


class ProcgenEnvFruitBotEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='fruitbot',
            distribution_mode='easy',
            param_values=[[-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-FruitBot-Easy-v0',
    entry_point=module_path + ':ProcgenEnvFruitBotEasy',
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenEnvFruitBotEasyTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='fruitbot',
            distribution_mode='easy',
            param_values=[[-1], [-1], [-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-FruitBot-Easy-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvFruitBotEasyTerminal',
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenEnvFruitBotHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='fruitbot',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-FruitBot-Hard-v0',
    entry_point=module_path + ':ProcgenEnvFruitBotHard',
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenEnvFruitBotHardTerminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='fruitbot',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1], [-1]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-FruitBot-Hard-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvFruitBotHardTerminal',
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
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
    "easy": FRUITBOT_EASY_PARAMS_P1,
    "hard": FRUITBOT_HARD_PARAMS_P1,
}

p2_params = {
    "easy": FRUITBOT_EASY_PARAMS_P2,
    "hard": FRUITBOT_HARD_PARAMS_P2,
}

p3_params = {
    "easy": FRUITBOT_EASY_PARAMS_P3,
    "hard": FRUITBOT_HARD_PARAMS_P3,
}

p4_params = {
    "hard": FRUITBOT_HARD_PARAMS_P4,
}

## Easy distribution
mode = distribution_modes_to_use[0]
assert mode == "easy"
p1_params_this_mode = p1_params[mode]
p2_params_this_mode = p2_params[mode]
p3_params_this_mode = p3_params[mode]
for rew in reward_modes_to_use:
    for act in action_spaces_to_use:
        for w in p1_params_this_mode:
            for g in p2_params_this_mode:
                for b in p3_params_this_mode:

                    mode_str = mode[0].upper() + mode[1:]
                    rew_str = ""
                    if rew == "terminal":
                        rew_str = "Terminal"
                    elif rew != "original":
                        raise NotImplementedError
                    if act != "original":
                        raise NotImplementedError

                    class_name = f"ProcgenEnvFruitBot{mode_str}W{w}to{w}G{g}to{g}B{b}to{b}{rew_str}"

                    class_definition_str = ""
                    class_definition_str += (
                        f"class {class_name}(ProcgenParamsEnv):\n"
                        f"    def __init__(self, seed=None, **kwargs):\n"
                        f"        super().__init__(\n"
                        f"            game='fruitbot',\n"
                        f"            distribution_mode='{mode}',\n"
                        f"            param_values=[[{w}], [{g}], [{b}]],\n"
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
                    gym_id += f"Procgen-FruitBot-{mode_str}-W{w}to{w}-G{g}to{g}-B{b}to{b}-"
                    if rew == "terminal":
                        gym_id += "Terminal-"
                    gym_id += "v0"

                    class_registration_str = (
                        f"gym_register(\n"
                        f"    id='{gym_id}',\n"
                        f"    entry_point=module_path + ':{class_name}',\n"
                        f"    max_episode_steps={FRUITBOT_EPISODE_STEPS},\n"
                        f")\n"
                        f"\n"
                    )
                    # print(class_registration_str)
                    eval(class_registration_str)

## Hard distribution
load_fruitbot_hard = False

if load_fruitbot_hard:
    mode = distribution_modes_to_use[1]
    assert mode == "hard"
    p1_params_this_mode = p1_params[mode]
    p2_params_this_mode = p2_params[mode]
    p3_params_this_mode = p3_params[mode]
    p4_params_this_mode = p4_params[mode]
    for rew in reward_modes_to_use:
        for act in action_spaces_to_use:
            for w in p1_params_this_mode:
                for g in p2_params_this_mode:
                    for b in p3_params_this_mode:
                        for l in p4_params_this_mode:

                            mode_str = mode[0].upper() + mode[1:]
                            rew_str = ""
                            if rew == "terminal":
                                rew_str = "Terminal"
                            elif rew != "original":
                                raise NotImplementedError
                            if act != "original":
                                raise NotImplementedError

                            class_name = f"ProcgenEnvFruitBot{mode_str}W{w}to{w}G{g}to{g}B{b}to{b}L{l}to{l}{rew_str}"

                            class_definition_str = ""
                            class_definition_str += (
                                f"class {class_name}(ProcgenParamsEnv):\n"
                                f"    def __init__(self, seed=None, **kwargs):\n"
                                f"        super().__init__(\n"
                                f"            game='fruitbot',\n"
                                f"            distribution_mode='{mode}',\n"
                                f"            param_values=[[{w}], [{g}], [{b}], [{l}]],\n"
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
                            gym_id += f"Procgen-FruitBot-{mode_str}-W{w}to{w}-G{g}to{g}-B{b}to{b}-L{l}to{l}-"
                            if rew == "terminal":
                                gym_id += "Terminal-"
                            gym_id += "v0"

                            class_registration_str = (
                                f"gym_register(\n"
                                f"    id='{gym_id}',\n"
                                f"    entry_point=module_path + ':{class_name}',\n"
                                f"    max_episode_steps={FRUITBOT_EPISODE_STEPS},\n"
                                f")\n"
                                f"\n"
                            )
                            # print(class_registration_str)
                            eval(class_registration_str)
