from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .maze import (
    MAZE_EPISODE_STEPS,
    MAZE_EASY_PARAMS_P1,
    MAZE_HARD_PARAMS_P1,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Maze: Easy distribution
class ProcgenMazeEasyD0to6Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="maze",
            distribution_mode="easy",
            param_values=[MAZE_EASY_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Maze-Easy-D0to6-Adversarial-v0",
    entry_point=module_path + ":ProcgenMazeEasyD0to6Adversarial",
    max_episode_steps=MAZE_EPISODE_STEPS,
)


class ProcgenMazeEasyD0to6MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="maze",
            distribution_mode="easy",
            param_values=[MAZE_EASY_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Maze-Easy-D0to6-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenMazeEasyD0to6MinParamAdversarial",
    max_episode_steps=MAZE_EPISODE_STEPS,
)


# Maze: Hard distribution
class ProcgenMazeHardD0to11Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="maze",
            distribution_mode="hard",
            param_values=[MAZE_HARD_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Maze-Hard-D0to11-Adversarial-v0",
    entry_point=module_path + ":ProcgenMazeHardD0to11Adversarial",
    max_episode_steps=MAZE_EPISODE_STEPS,
)


class ProcgenMazeHardD0to11MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="maze",
            distribution_mode="hard",
            param_values=[MAZE_HARD_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Maze-Hard-D0to11-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenMazeHardD0to11MinParamAdversarial",
    max_episode_steps=MAZE_EPISODE_STEPS,
)
