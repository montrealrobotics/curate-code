from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .climber import (
    CLIMBER_EPISODE_STEPS,
    CLIMBER_EASY_PARAMS_P1,
    CLIMBER_EASY_PARAMS_P2,
    CLIMBER_HARD_PARAMS_P1,
    CLIMBER_HARD_PARAMS_P2,
    CLIMBER_EASIER_PARAMS_P1,
    CLIMBER_EASIER_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Climber: Easy distribution mode
class ProcgenClimberEasyP1to10E0to20Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASY_PARAMS_P1, CLIMBER_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to10-E0to20-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to10E0to20Adversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to10E0to20MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASY_PARAMS_P1, CLIMBER_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to10-E0to20-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to10E0to20MinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to10E0to20TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASY_PARAMS_P1, CLIMBER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to10-E0to20-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to10E0to20TerminalAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to10E0to20TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASY_PARAMS_P1, CLIMBER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to10-E0to20-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenClimberEasyP1to10E0to20TerminalMinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


# Climber: Hard distribution mode
class ProcgenClimberHardP1to10E0to50Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="hard",
            param_values=[CLIMBER_HARD_PARAMS_P1, CLIMBER_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Hard-P1to10-E0to50-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberHardP1to10E0to50Adversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberHardP1to10E0to50MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="hard",
            param_values=[CLIMBER_HARD_PARAMS_P1, CLIMBER_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Hard-P1to10-E0to50-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberHardP1to10E0to50MinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberHardP1to10E0to50TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="hard",
            param_values=[CLIMBER_HARD_PARAMS_P1, CLIMBER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Hard-P1to10-E0to50-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberHardP1to10E0to50TerminalAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberHardP1to10E0to50TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="hard",
            param_values=[CLIMBER_HARD_PARAMS_P1, CLIMBER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Hard-P1to10-E0to50-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenClimberHardP1to10E0to50TerminalMinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


# Climber: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenClimberEasyP1to5E0to20Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASIER_PARAMS_P1, CLIMBER_EASIER_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to5-E0to20-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to5E0to20Adversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to5E0to20MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASIER_PARAMS_P1, CLIMBER_EASIER_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to5-E0to20-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to5E0to20MinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to5E0to20TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASIER_PARAMS_P1, CLIMBER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to5-E0to20-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenClimberEasyP1to5E0to20TerminalAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)


class ProcgenClimberEasyP1to5E0to20TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="climber",
            distribution_mode="easy",
            param_values=[CLIMBER_EASIER_PARAMS_P1, CLIMBER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Climber-Easy-P1to5-E0to20-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenClimberEasyP1to5E0to20TerminalMinParamAdversarial",
    max_episode_steps=CLIMBER_EPISODE_STEPS,
)
