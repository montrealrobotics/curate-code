from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .ninja import (
    NINJA_EPISODE_STEPS,
    NINJA_EASY_PARAMS_P1,
    NINJA_EASY_PARAMS_P2,
    NINJA_HARD_PARAMS_P1,
    NINJA_HARD_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Ninja: Easy distribution mode
class ProcgenNinjaEasyD1to3S1to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="ninja",
            distribution_mode="easy",
            param_values=[NINJA_EASY_PARAMS_P1, NINJA_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Ninja-Easy-D1to3-S1to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenNinjaEasyD1to3S1to5Adversarial",
    max_episode_steps=NINJA_EPISODE_STEPS,
)


class ProcgenNinjaEasyD1to3S1to5MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="ninja",
            distribution_mode="easy",
            param_values=[NINJA_EASY_PARAMS_P1, NINJA_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Ninja-Easy-D1to3-S1to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenNinjaEasyD1to3S1to5MinParamAdversarial",
    max_episode_steps=NINJA_EPISODE_STEPS,
)


# Ninja: Hard distribution mode
class ProcgenNinjaHardD1to3S1to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="ninja",
            distribution_mode="hard",
            param_values=[NINJA_HARD_PARAMS_P1, NINJA_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Ninja-Hard-D1to3-S1to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenNinjaHardD1to3S1to5Adversarial",
    max_episode_steps=NINJA_EPISODE_STEPS,
)


class ProcgenNinjaHardD1to3S1to5MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="ninja",
            distribution_mode="hard",
            param_values=[NINJA_HARD_PARAMS_P1, NINJA_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Ninja-Hard-D1to3-S1to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenNinjaHardD1to3S1to5MinParamAdversarial",
    max_episode_steps=NINJA_EPISODE_STEPS,
)
