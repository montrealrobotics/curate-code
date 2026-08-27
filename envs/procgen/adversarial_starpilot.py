from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .starpilot import (
    STARPILOT_EPISODE_STEPS,
    STARPILOT_EASY_PARAMS_P1,
    STARPILOT_EASY_PARAMS_P2,
    STARPILOT_EASY_PARAMS_P3,
    STARPILOT_EASY_PARAMS_P4,
    STARPILOT_HARD_PARAMS_P1,
    STARPILOT_HARD_PARAMS_P2,
    STARPILOT_HARD_PARAMS_P3,
    STARPILOT_HARD_PARAMS_P4,
    STARPILOT_EASIER_PARAMS_P1,
    STARPILOT_EASIER_PARAMS_P2,
    STARPILOT_EASIER_PARAMS_P3,
    STARPILOT_EASIER_PARAMS_P4,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# StarPilot: Easy distribution mode
class ProcgenStarPilotEasyW1to500T1to20G1to5F1to90Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASY_PARAMS_P1,
                STARPILOT_EASY_PARAMS_P2,
                STARPILOT_EASY_PARAMS_P3,
                STARPILOT_EASY_PARAMS_P4,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to500-T1to20-G1to5-F1to90-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to500T1to20G1to5F1to90Adversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to500T1to20G1to5F1to90MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASY_PARAMS_P1,
                STARPILOT_EASY_PARAMS_P2,
                STARPILOT_EASY_PARAMS_P3,
                STARPILOT_EASY_PARAMS_P4,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to500-T1to20-G1to5-F1to90-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to500T1to20G1to5F1to90MinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to500T1to20G1to5F1to90TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASY_PARAMS_P1,
                STARPILOT_EASY_PARAMS_P2,
                STARPILOT_EASY_PARAMS_P3,
                STARPILOT_EASY_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to500-T1to20-G1to5-F1to90-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to500T1to20G1to5F1to90TerminalAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to500T1to20G1to5F1to90TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASY_PARAMS_P1,
                STARPILOT_EASY_PARAMS_P2,
                STARPILOT_EASY_PARAMS_P3,
                STARPILOT_EASY_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to500-T1to20-G1to5-F1to90-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to500T1to20G1to5F1to90TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


# StarPilot: Hard distribution mode
class ProcgenStarPilotHardW1to500T1to20G1to5F1to90Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_HARD_PARAMS_P1,
                STARPILOT_HARD_PARAMS_P2,
                STARPILOT_HARD_PARAMS_P3,
                STARPILOT_HARD_PARAMS_P4,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Hard-W1to500-T1to20-G1to5-F1to90-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotHardW1to500T1to20G1to5F1to90Adversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotHardW1to500T1to20G1to5F1to90MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_HARD_PARAMS_P1,
                STARPILOT_HARD_PARAMS_P2,
                STARPILOT_HARD_PARAMS_P3,
                STARPILOT_HARD_PARAMS_P4,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Hard-W1to500-T1to20-G1to5-F1to90-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotHardW1to500T1to20G1to5F1to90MinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotHardW1to500T1to20G1to5F1to90TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_HARD_PARAMS_P1,
                STARPILOT_HARD_PARAMS_P2,
                STARPILOT_HARD_PARAMS_P3,
                STARPILOT_HARD_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Hard-W1to500-T1to20-G1to5-F1to90-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotHardW1to500T1to20G1to5F1to90TerminalAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotHardW1to500T1to20G1to5F1to90TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="hard",
            param_values=[
                STARPILOT_HARD_PARAMS_P1,
                STARPILOT_HARD_PARAMS_P2,
                STARPILOT_HARD_PARAMS_P3,
                STARPILOT_HARD_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Hard-W1to500-T1to20-G1to5-F1to90-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotHardW1to500T1to20G1to5F1to90TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


# StarPilot: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenStarPilotEasyW1to250T1to10G1to3F1to45Adversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASIER_PARAMS_P1,
                STARPILOT_EASIER_PARAMS_P2,
                STARPILOT_EASIER_PARAMS_P3,
                STARPILOT_EASIER_PARAMS_P4,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to250-T1to10-G1to3-F1to45-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to250T1to10G1to3F1to45Adversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to250T1to10G1to3F1to45MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASIER_PARAMS_P1,
                STARPILOT_EASIER_PARAMS_P2,
                STARPILOT_EASIER_PARAMS_P3,
                STARPILOT_EASIER_PARAMS_P4,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to250-T1to10-G1to3-F1to45-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to250T1to10G1to3F1to45MinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to250T1to10G1to3F1to45TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASIER_PARAMS_P1,
                STARPILOT_EASIER_PARAMS_P2,
                STARPILOT_EASIER_PARAMS_P3,
                STARPILOT_EASIER_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to250-T1to10-G1to3-F1to45-Terminal-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to250T1to10G1to3F1to45TerminalAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenStarPilotEasyW1to250T1to10G1to3F1to45TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="starpilot",
            distribution_mode="easy",
            param_values=[
                STARPILOT_EASIER_PARAMS_P1,
                STARPILOT_EASIER_PARAMS_P2,
                STARPILOT_EASIER_PARAMS_P3,
                STARPILOT_EASIER_PARAMS_P4,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-StarPilot-Easy-W1to250-T1to10-G1to3-F1to45-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenStarPilotEasyW1to250T1to10G1to3F1to45TerminalMinParamAdversarial",
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)
