from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .bigfish import (
    BIGFISH_EPISODE_STEPS,
    BIGFISH_EASY_PARAMS_P1,
    BIGFISH_HARD_PARAMS_P1,
    BIGFISH_EASIER_PARAMS_P1,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# BigFish: Easy distribution mode
class ProcgenBigFishEasyF1to30Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASY_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to30-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to30Adversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to30MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASY_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to30-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to30MinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to30TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to30-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to30TerminalAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to30TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to30-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to30TerminalMinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


# BigFish: Hard distribution mode
class ProcgenBigFishHardF1to30Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="hard",
            param_values=[BIGFISH_HARD_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Hard-F1to30-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishHardF1to30Adversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishHardF1to30MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="hard",
            param_values=[BIGFISH_HARD_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Hard-F1to30-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishHardF1to30MinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishHardF1to30TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="hard",
            param_values=[BIGFISH_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Hard-F1to30-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishHardF1to30TerminalAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishHardF1to30TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="hard",
            param_values=[BIGFISH_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Hard-F1to30-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishHardF1to30TerminalMinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


# BigFish: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenBigFishEasyF1to10Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASIER_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to10-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to10Adversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to10MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASIER_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to10-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to10MinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to10TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASIER_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to10-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to10TerminalAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)


class ProcgenBigFishEasyF1to10TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bigfish",
            distribution_mode="easy",
            param_values=[BIGFISH_EASIER_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BigFish-Easy-F1to10-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBigFishEasyF1to10TerminalMinParamAdversarial",
    max_episode_steps=BIGFISH_EPISODE_STEPS,
)
