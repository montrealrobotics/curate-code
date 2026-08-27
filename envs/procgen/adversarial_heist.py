from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .heist import (
    HEIST_EPISODE_STEPS,
    HEIST_EASY_PARAMS_P1,
    HEIST_EASY_PARAMS_P2,
    HEIST_HARD_PARAMS_P1,
    HEIST_HARD_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Heist: Easy distribution mode
class ProcgenHeistEasyD0to2K0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="heist",
            distribution_mode="easy",
            param_values=[HEIST_EASY_PARAMS_P1, HEIST_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Heist-Easy-D0to2-K0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenHeistEasyD0to2K0to3Adversarial",
    max_episode_steps=HEIST_EPISODE_STEPS,
)


class ProcgenHeistEasyD0to2K0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="heist",
            distribution_mode="easy",
            param_values=[HEIST_EASY_PARAMS_P1, HEIST_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Heist-Easy-D0to2-K0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenHeistEasyD0to2K0to3MinParamAdversarial",
    max_episode_steps=HEIST_EPISODE_STEPS,
)


# Heist: Hard distribution mode
class ProcgenHeistHardD0to4K0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="heist",
            distribution_mode="hard",
            param_values=[HEIST_HARD_PARAMS_P1, HEIST_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Heist-Hard-D0to4-K0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenHeistHardD0to4K0to3Adversarial",
    max_episode_steps=HEIST_EPISODE_STEPS,
)


class ProcgenHeistHardD0to4K0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="heist",
            distribution_mode="hard",
            param_values=[HEIST_HARD_PARAMS_P1, HEIST_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Heist-Hard-D0to4-K0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenHeistHardD0to4K0to3MinParamAdversarial",
    max_episode_steps=HEIST_EPISODE_STEPS,
)
