from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .jumper import (
    JUMPER_EPISODE_STEPS,
    JUMPER_EASY_PARAMS_P1,
    JUMPER_HARD_PARAMS_P1,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Jumper: Easy distribution
class ProcgenJumperEasyS0to20Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="jumper",
            distribution_mode="easy",
            param_values=[JUMPER_EASY_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Jumper-Easy-S0to20-Adversarial-v0",
    entry_point=module_path + ":ProcgenJumperEasyS0to20Adversarial",
    max_episode_steps=JUMPER_EPISODE_STEPS,
)


class ProcgenJumperEasyS0to20MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="jumper",
            distribution_mode="easy",
            param_values=[JUMPER_EASY_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Jumper-Easy-S0to20-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenJumperEasyS0to20MinParamAdversarial",
    max_episode_steps=JUMPER_EPISODE_STEPS,
)


# Jumper: Hard distribution
class ProcgenJumperHardS0to20Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="jumper",
            distribution_mode="hard",
            param_values=[JUMPER_HARD_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Jumper-Hard-S0to20-Adversarial-v0",
    entry_point=module_path + ":ProcgenJumperHardS0to20Adversarial",
    max_episode_steps=JUMPER_EPISODE_STEPS,
)


class ProcgenJumperHardS0to20MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="jumper",
            distribution_mode="hard",
            param_values=[JUMPER_HARD_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Jumper-Hard-S0to20-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenJumperHardS0to20MinParamAdversarial",
    max_episode_steps=JUMPER_EPISODE_STEPS,
)
