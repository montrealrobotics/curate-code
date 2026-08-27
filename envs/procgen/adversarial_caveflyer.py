from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .caveflyer import (
    CAVEFLYER_EPISODE_STEPS,
    CAVEFLYER_EASY_PARAMS_P1,
    CAVEFLYER_HARD_PARAMS_P1,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# CaveFlyer: Easy distribution mode
class ProcgenCaveFlyerEasyO0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="easy",
            param_values=[CAVEFLYER_EASY_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Easy-O0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerEasyO0to3Adversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerEasyO0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="easy",
            param_values=[CAVEFLYER_EASY_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Easy-O0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerEasyO0to3MinParamAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerEasyO0to3TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="easy",
            param_values=[CAVEFLYER_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Easy-O0to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerEasyO0to3TerminalAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerEasyO0to3TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="easy",
            param_values=[CAVEFLYER_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Easy-O0to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerEasyO0to3TerminalMinParamAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


# CaveFlyer: Hard distribution mode
class ProcgenCaveFlyerHardO0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="hard",
            param_values=[CAVEFLYER_HARD_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Hard-O0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerHardO0to3Adversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerHardO0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="hard",
            param_values=[CAVEFLYER_HARD_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Hard-O0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerHardO0to3MinParamAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerHardO0to3TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="hard",
            param_values=[CAVEFLYER_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Hard-O0to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerHardO0to3TerminalAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)


class ProcgenCaveFlyerHardO0to3TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="caveflyer",
            distribution_mode="hard",
            param_values=[CAVEFLYER_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-CaveFlyer-Hard-O0to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenCaveFlyerHardO0to3TerminalMinParamAdversarial",
    max_episode_steps=CAVEFLYER_EPISODE_STEPS,
)
