from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


MAZE_EPISODE_STEPS = 500
MAZE_EASY_PARAMS_P1 = list(range(6+1))
MAZE_HARD_PARAMS_P1 = list(range(11+1))


class ProcgenEnvMazeEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='maze',
            distribution_mode='easy',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )

gym_register(
    id='Procgen-Maze-Easy-v0',
    entry_point=module_path + ':ProcgenEnvMazeEasy',
    max_episode_steps=MAZE_EPISODE_STEPS,
)


class ProcgenEnvMazeHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='maze',
            distribution_mode='hard',
            param_values=[[-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-Maze-Hard-v0',
    entry_point=module_path + ':ProcgenEnvMazeHard',
    max_episode_steps=MAZE_EPISODE_STEPS,
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
    "easy": MAZE_EASY_PARAMS_P1,
    "hard": MAZE_HARD_PARAMS_P1,
}

##
for mode in distribution_modes_to_use:
    p1_params_this_mode = p1_params[mode]
    for act in action_spaces_to_use:
        for d in p1_params_this_mode:

            mode_str = mode[0].upper() + mode[1:]
            act_str = ""
            if act != "original":
                raise NotImplementedError

            class_name = f"ProcgenEnvMaze{mode_str}D{d}to{d}"

            class_definition_str = ""
            class_definition_str += (
                f"class {class_name}(ProcgenParamsEnv):\n"
                f"    def __init__(self, seed=None, **kwargs):\n"
                f"        super().__init__(\n"
                f"            game='maze',\n"
                f"            distribution_mode='{mode}',\n"
                f"            param_values=[[{d}]],\n"
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
            gym_id += f"Procgen-Maze-{mode_str}-D{d}to{d}-"
            gym_id += "v0"

            class_registration_str = (
                f"gym_register(\n"
                f"    id='{gym_id}',\n"
                f"    entry_point=module_path + ':{class_name}',\n"
                f"    max_episode_steps={MAZE_EPISODE_STEPS},\n"
                f")\n"
                f"\n"
            )
            # print(class_registration_str)
            eval(class_registration_str)
