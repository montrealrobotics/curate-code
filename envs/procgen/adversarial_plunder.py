from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .plunder import (
    PLUNDER_EPISODE_STEPS,
    PLUNDER_EASY_PARAMS_P1,
    PLUNDER_EASY_PARAMS_P2,
    PLUNDER_HARD_PARAMS_P1,
    PLUNDER_HARD_PARAMS_P2,
    PLUNDER_HARD_PARAMS_P3,
    PLUNDER_EASIER_PARAMS_P1,
    PLUNDER_EASIER_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Plunder: Easy distribution mode
class ProcgenPlunderEasyT1to20J1to10Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASY_PARAMS_P1, PLUNDER_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to20-J1to10-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to20J1to10Adversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to20J1to10MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASY_PARAMS_P1, PLUNDER_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to20-J1to10-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to20J1to10MinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to20J1to10TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASY_PARAMS_P1, PLUNDER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to20-J1to10-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to20J1to10TerminalAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to20J1to10TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASY_PARAMS_P1, PLUNDER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to20-J1to10-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenPlunderEasyT1to20J1to10TerminalMinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


# Plunder: Hard distribution mode
class ProcgenPlunderHardT1to20J1to10P0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="hard",
            param_values=[
                PLUNDER_HARD_PARAMS_P1,
                PLUNDER_HARD_PARAMS_P2,
                PLUNDER_HARD_PARAMS_P3,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Hard-T1to20-J1to10-P0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderHardT1to20J1to10P0to3Adversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderHardT1to20J1to10P0to3MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="hard",
            param_values=[
                PLUNDER_HARD_PARAMS_P1,
                PLUNDER_HARD_PARAMS_P2,
                PLUNDER_HARD_PARAMS_P3,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Hard-T1to20-J1to10-P0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderHardT1to20J1to10P0to3MinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderHardT1to20J1to10P0to3TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="hard",
            param_values=[
                PLUNDER_HARD_PARAMS_P1,
                PLUNDER_HARD_PARAMS_P2,
                PLUNDER_HARD_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Hard-T1to20-J1to10-P0to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderHardT1to20J1to10P0to3TerminalAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderHardT1to20J1to10P0to3TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="hard",
            param_values=[
                PLUNDER_HARD_PARAMS_P1,
                PLUNDER_HARD_PARAMS_P2,
                PLUNDER_HARD_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Hard-T1to20-J1to10-P0to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenPlunderHardT1to20J1to10P0to3TerminalMinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


# Plunder: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenPlunderEasyT1to8J1to10Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASIER_PARAMS_P1, PLUNDER_EASIER_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to8-J1to10-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to8J1to10Adversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to8J1to10MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASIER_PARAMS_P1, PLUNDER_EASIER_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to8-J1to10-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to8J1to10MinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to8J1to10TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASIER_PARAMS_P1, PLUNDER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to8-J1to10-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenPlunderEasyT1to8J1to10TerminalAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)


class ProcgenPlunderEasyT1to8J1to10TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="plunder",
            distribution_mode="easy",
            param_values=[PLUNDER_EASIER_PARAMS_P1, PLUNDER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Plunder-Easy-T1to8-J1to10-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenPlunderEasyT1to8J1to10TerminalMinParamAdversarial",
    max_episode_steps=PLUNDER_EPISODE_STEPS,
)
