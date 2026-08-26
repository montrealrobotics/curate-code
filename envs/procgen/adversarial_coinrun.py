from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .coinrun import (
    COINRUN_EPISODE_STEPS,
    COINRUN_EASY_PARAMS_P1,
    COINRUN_EASY_PARAMS_P2,
    COINRUN_HARD_PARAMS_P1,
    COINRUN_HARD_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# CoinRun: Easy distribution mode
class ProcgenCoinRunEasyD1to3S1to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="coinrun",
            distribution_mode="easy",
            param_values=[COINRUN_EASY_PARAMS_P1, COINRUN_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CoinRun-Easy-D1to3-S1to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenCoinRunEasyD1to3S1to5Adversarial",
    max_episode_steps=COINRUN_EPISODE_STEPS,
)


class ProcgenCoinRunEasyD1to3S1to5MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="coinrun",
            distribution_mode="easy",
            param_values=[COINRUN_EASY_PARAMS_P1, COINRUN_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CoinRun-Easy-D1to3-S1to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCoinRunEasyD1to3S1to5MinParamAdversarial",
    max_episode_steps=COINRUN_EPISODE_STEPS,
)


# CoinRun: Hard distribution mode
class ProcgenCoinRunHardD1to3S1to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="coinrun",
            distribution_mode="hard",
            param_values=[COINRUN_HARD_PARAMS_P1, COINRUN_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CoinRun-Hard-D1to3-S1to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenCoinRunHardD1to3S1to5Adversarial",
    max_episode_steps=COINRUN_EPISODE_STEPS,
)


class ProcgenCoinRunHardD1to3S1to5MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="coinrun",
            distribution_mode="hard",
            param_values=[COINRUN_HARD_PARAMS_P1, COINRUN_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CoinRun-Hard-D1to3-S1to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCoinRunHardD1to3S1to5MinParamAdversarial",
    max_episode_steps=COINRUN_EPISODE_STEPS,
)
