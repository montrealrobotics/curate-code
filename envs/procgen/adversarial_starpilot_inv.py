from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .starpilot_inv import (
    STARPILOT_INV_EPISODE_STEPS,
    STARPILOT_INV_EASY_PARAMS_P1,
    STARPILOT_INV_EASY_PARAMS_P2,
    STARPILOT_INV_EASY_PARAMS_P3,
    STARPILOT_INV_EASY_PARAMS_P4,
    STARPILOT_INV_HARD_PARAMS_P1,
    STARPILOT_INV_HARD_PARAMS_P2,
    STARPILOT_INV_HARD_PARAMS_P3,
    STARPILOT_INV_HARD_PARAMS_P4,
    STARPILOT_INV_EASIER_PARAMS_P1,
    STARPILOT_INV_EASIER_PARAMS_P2,
    STARPILOT_INV_EASIER_PARAMS_P3,
    STARPILOT_INV_EASIER_PARAMS_P4,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# StarPilotInv: Easy distribution mode
class ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASY_PARAMS_P1,
                STARPILOT_INV_EASY_PARAMS_P2,
                STARPILOT_INV_EASY_PARAMS_P3,
                STARPILOT_INV_EASY_PARAMS_P4,
            ],
            level_options_mode=1,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to500-T1to20-G1to5-F1to90-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90Adversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASY_PARAMS_P1,
                STARPILOT_INV_EASY_PARAMS_P2,
                STARPILOT_INV_EASY_PARAMS_P3,
                STARPILOT_INV_EASY_PARAMS_P4,
            ],
            level_options_mode=1,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to500-T1to20-G1to5-F1to90-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90MinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASY_PARAMS_P1,
                STARPILOT_INV_EASY_PARAMS_P2,
                STARPILOT_INV_EASY_PARAMS_P3,
                STARPILOT_INV_EASY_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to500-T1to20-G1to5-F1to90-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90TerminalAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASY_PARAMS_P1,
                STARPILOT_INV_EASY_PARAMS_P2,
                STARPILOT_INV_EASY_PARAMS_P3,
                STARPILOT_INV_EASY_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to500-T1to20-G1to5-F1to90-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to500T1to20G1to5F1to90TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


# StarPilotInv: Hard distribution mode
class ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_INV_HARD_PARAMS_P1,
                STARPILOT_INV_HARD_PARAMS_P2,
                STARPILOT_INV_HARD_PARAMS_P3,
                STARPILOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Hard-W1to500-T1to20-G1to5-F1to90-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90Adversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_INV_HARD_PARAMS_P1,
                STARPILOT_INV_HARD_PARAMS_P2,
                STARPILOT_INV_HARD_PARAMS_P3,
                STARPILOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Hard-W1to500-T1to20-G1to5-F1to90-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90MinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_INV_HARD_PARAMS_P1,
                STARPILOT_INV_HARD_PARAMS_P2,
                STARPILOT_INV_HARD_PARAMS_P3,
                STARPILOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Hard-W1to500-T1to20-G1to5-F1to90-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90TerminalAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_INV_HARD_PARAMS_P1,
                STARPILOT_INV_HARD_PARAMS_P2,
                STARPILOT_INV_HARD_PARAMS_P3,
                STARPILOT_INV_HARD_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Hard-W1to500-T1to20-G1to5-F1to90-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvHardW1to500T1to20G1to5F1to90TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


# StarPilotInv: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASIER_PARAMS_P1,
                STARPILOT_INV_EASIER_PARAMS_P2,
                STARPILOT_INV_EASIER_PARAMS_P3,
                STARPILOT_INV_EASIER_PARAMS_P4,
            ],
            level_options_mode=1,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to250-T11to20-G1to3-F46to90-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90Adversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASIER_PARAMS_P1,
                STARPILOT_INV_EASIER_PARAMS_P2,
                STARPILOT_INV_EASIER_PARAMS_P3,
                STARPILOT_INV_EASIER_PARAMS_P4,
            ],
            level_options_mode=1,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to250-T11to20-G1to3-F46to90-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90MinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASIER_PARAMS_P1,
                STARPILOT_INV_EASIER_PARAMS_P2,
                STARPILOT_INV_EASIER_PARAMS_P3,
                STARPILOT_INV_EASIER_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to250-T11to20-G1to3-F46to90-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90TerminalAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)


class ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_INV_EASIER_PARAMS_P1,
                STARPILOT_INV_EASIER_PARAMS_P2,
                STARPILOT_INV_EASIER_PARAMS_P3,
                STARPILOT_INV_EASIER_PARAMS_P4,
            ],
            level_options_mode=1,
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilotInv-Easy-W1to250-T11to20-G1to3-F46to90-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotInvEasyW1to250T11to20G1to3F46to90TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_INV_EPISODE_STEPS,
)
