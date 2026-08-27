from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .fruitbot import (
    FRUITBOT_EPISODE_STEPS,
    FRUITBOT_EASY_PARAMS_P1,
    FRUITBOT_EASY_PARAMS_P2,
    FRUITBOT_EASY_PARAMS_P3,
    FRUITBOT_HARD_PARAMS_P1,
    FRUITBOT_HARD_PARAMS_P2,
    FRUITBOT_HARD_PARAMS_P3,
    FRUITBOT_HARD_PARAMS_P4,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# FruitBot: Easy distribution mode
class ProcgenFruitBotEasyW1to5G0to60B0to10Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_EASY_PARAMS_P1,
                FRUITBOT_EASY_PARAMS_P2,
                FRUITBOT_EASY_PARAMS_P3,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Easy-W1to5-G0to60-B0to10-Adversarial-v0",
    entry_point=module_path + ":ProcgenFruitBotEasyW1to5G0to60B0to10Adversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotEasyW1to5G0to60B0to10MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_EASY_PARAMS_P1,
                FRUITBOT_EASY_PARAMS_P2,
                FRUITBOT_EASY_PARAMS_P3,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Easy-W1to5-G0to60-B0to10-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotEasyW1to5G0to60B0to10MinParamAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotEasyW1to5G0to60B0to10TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_EASY_PARAMS_P1,
                FRUITBOT_EASY_PARAMS_P2,
                FRUITBOT_EASY_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Easy-W1to5-G0to60-B0to10-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotEasyW1to5G0to60B0to10TerminalAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotEasyW1to5G0to60B0to10TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_EASY_PARAMS_P1,
                FRUITBOT_EASY_PARAMS_P2,
                FRUITBOT_EASY_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Easy-W1to5-G0to60-B0to10-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotEasyW1to5G0to60B0to10TerminalMinParamAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


# FruitBot: Hard distribution mode
class ProcgenFruitBotHardW1to10G0to70B0to10L0to5Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_HARD_PARAMS_P1,
                FRUITBOT_HARD_PARAMS_P2,
                FRUITBOT_HARD_PARAMS_P3,
                FRUITBOT_HARD_PARAMS_P4,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Hard-W1to10-G0to70-B0to10-L0to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenFruitBotHardW1to10G0to70B0to10L0to5Adversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotHardW1to10G0to70B0to10L0to5MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_HARD_PARAMS_P1,
                FRUITBOT_HARD_PARAMS_P2,
                FRUITBOT_HARD_PARAMS_P3,
                FRUITBOT_HARD_PARAMS_P4,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Hard-W1to10-G0to70-B0to10-L0to5-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotHardW1to10G0to70B0to10L0to5MinParamAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotHardW1to10G0to70B0to10L0to5TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_HARD_PARAMS_P1,
                FRUITBOT_HARD_PARAMS_P2,
                FRUITBOT_HARD_PARAMS_P3,
                FRUITBOT_HARD_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Hard-W1to10-G0to70-B0to10-L0to5-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotHardW1to10G0to70B0to10L0to5TerminalAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)


class ProcgenFruitBotHardW1to10G0to70B0to10L0to5TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_HARD_PARAMS_P1,
                FRUITBOT_HARD_PARAMS_P2,
                FRUITBOT_HARD_PARAMS_P3,
                FRUITBOT_HARD_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBot-Hard-W1to10-G0to70-B0to10-L0to5-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotHardW1to10G0to70B0to10L0to5TerminalMinParamAdversarial",
    max_episode_steps=FRUITBOT_EPISODE_STEPS,
)
