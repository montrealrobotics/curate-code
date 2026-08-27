from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .fruitbot_inv import (
    FRUITBOT_INV_EPISODE_STEPS,
    FRUITBOT_INV_EASY_PARAMS_P1,
    FRUITBOT_INV_EASY_PARAMS_P2,
    FRUITBOT_INV_EASY_PARAMS_P3,
    FRUITBOT_INV_HARD_PARAMS_P1,
    FRUITBOT_INV_HARD_PARAMS_P2,
    FRUITBOT_INV_HARD_PARAMS_P3,
    FRUITBOT_INV_HARD_PARAMS_P4,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# FruitBotInv: Easy distribution mode
class ProcgenFruitBotInvEasyW1to5G0to60B0to10Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_INV_EASY_PARAMS_P1,
                FRUITBOT_INV_EASY_PARAMS_P2,
                FRUITBOT_INV_EASY_PARAMS_P3,
            ],
            level_options_mode=1,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Easy-W1to5-G0to60-B0to10-Adversarial-v0",
    entry_point=module_path + ":ProcgenFruitBotInvEasyW1to5G0to60B0to10Adversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvEasyW1to5G0to60B0to10MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_INV_EASY_PARAMS_P1,
                FRUITBOT_INV_EASY_PARAMS_P2,
                FRUITBOT_INV_EASY_PARAMS_P3,
            ],
            level_options_mode=1,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Easy-W1to5-G0to60-B0to10-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvEasyW1to5G0to60B0to10MinParamAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvEasyW1to5G0to60B0to10TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_INV_EASY_PARAMS_P1,
                FRUITBOT_INV_EASY_PARAMS_P2,
                FRUITBOT_INV_EASY_PARAMS_P3,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Easy-W1to5-G0to60-B0to10-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvEasyW1to5G0to60B0to10TerminalAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvEasyW1to5G0to60B0to10TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="easy",
            param_values=[
                FRUITBOT_INV_EASY_PARAMS_P1,
                FRUITBOT_INV_EASY_PARAMS_P2,
                FRUITBOT_INV_EASY_PARAMS_P3,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Easy-W1to5-G0to60-B0to10-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvEasyW1to5G0to60B0to10TerminalMinParamAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


# FruitBotInv: Hard distribution mode
class ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_INV_HARD_PARAMS_P1,
                FRUITBOT_INV_HARD_PARAMS_P2,
                FRUITBOT_INV_HARD_PARAMS_P3,
                FRUITBOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Hard-W1to10-G0to70-B0to10-L0to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5Adversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_INV_HARD_PARAMS_P1,
                FRUITBOT_INV_HARD_PARAMS_P2,
                FRUITBOT_INV_HARD_PARAMS_P3,
                FRUITBOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Hard-W1to10-G0to70-B0to10-L0to5-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5MinParamAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_INV_HARD_PARAMS_P1,
                FRUITBOT_INV_HARD_PARAMS_P2,
                FRUITBOT_INV_HARD_PARAMS_P3,
                FRUITBOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Hard-W1to10-G0to70-B0to10-L0to5-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5TerminalAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)


class ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="fruitbot",
            distribution_mode="hard",
            param_values=[
                FRUITBOT_INV_HARD_PARAMS_P1,
                FRUITBOT_INV_HARD_PARAMS_P2,
                FRUITBOT_INV_HARD_PARAMS_P3,
                FRUITBOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-FruitBotInv-Hard-W1to10-G0to70-B0to10-L0to5-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenFruitBotInvHardW1to10G0to70B0to10L0to5TerminalMinParamAdversarial",
    max_episode_steps=FRUITBOT_INV_EPISODE_STEPS,
)
