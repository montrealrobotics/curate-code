from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .leaper import (
    LEAPER_EPISODE_STEPS,
    LEAPER_EASY_PARAMS_P1,
    LEAPER_EASY_PARAMS_P2,
    LEAPER_HARD_PARAMS_P1,
    LEAPER_HARD_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Leaper: Easy distribution mode
class ProcgenLeaperEasyR0to3W0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="leaper",
            distribution_mode="easy",
            param_values=[LEAPER_EASY_PARAMS_P1, LEAPER_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Leaper-Easy-R0to3-W0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenLeaperEasyR0to3W0to3Adversarial",
    max_episode_steps=LEAPER_EPISODE_STEPS,
)


class ProcgenLeaperEasyR0to3W0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="leaper",
            distribution_mode="easy",
            param_values=[LEAPER_EASY_PARAMS_P1, LEAPER_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Leaper-Easy-R0to3-W0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenLeaperEasyR0to3W0to3MinParamAdversarial",
    max_episode_steps=LEAPER_EPISODE_STEPS,
)


# Leaper: Hard distribution mode
class ProcgenLeaperHardR0to5W0to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="leaper",
            distribution_mode="hard",
            param_values=[LEAPER_HARD_PARAMS_P1, LEAPER_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Leaper-Hard-R0to5-W0to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenLeaperHardR0to5W0to5Adversarial",
    max_episode_steps=LEAPER_EPISODE_STEPS,
)


class ProcgenLeaperHardR0to5W0to5MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="leaper",
            distribution_mode="hard",
            param_values=[LEAPER_HARD_PARAMS_P1, LEAPER_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Leaper-Hard-R0to5-W0to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenLeaperHardR0to5W0to5MinParamAdversarial",
    max_episode_steps=LEAPER_EPISODE_STEPS,
)
