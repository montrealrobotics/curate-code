from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .dodgeball import (
    DODGEBALL_EPISODE_STEPS,
    DODGEBALL_EASY_PARAMS_P1,
    DODGEBALL_HARD_PARAMS_P1,
    DODGEBALL_EASIER_PARAMS_P1,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Dodgeball: Easy distribution mode
class ProcgenDodgeballEasyE3to6Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASY_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E3to6-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE3to6Adversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE3to6MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASY_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E3to6-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE3to6MinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE3to6TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E3to6-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE3to6TerminalAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE3to6TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASY_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E3to6-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE3to6TerminalMinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


# Dodgeball: Hard distribution mode
class ProcgenDodgeballHardE3to6Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="hard",
            param_values=[DODGEBALL_HARD_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Hard-E3to6-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballHardE3to6Adversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballHardE3to6MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="hard",
            param_values=[DODGEBALL_HARD_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Hard-E3to6-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballHardE3to6MinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballHardE3to6TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="hard",
            param_values=[DODGEBALL_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Hard-E3to6-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballHardE3to6TerminalAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballHardE3to6TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="hard",
            param_values=[DODGEBALL_HARD_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Hard-E3to6-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballHardE3to6TerminalMinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


# Dodgeball: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenDodgeballEasyE0to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASIER_PARAMS_P1],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E0to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE0to3Adversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE0to3MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASIER_PARAMS_P1],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E0to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE0to3MinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE0to3TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASIER_PARAMS_P1],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E0to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE0to3TerminalAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)


class ProcgenDodgeballEasyE0to3TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="dodgeball",
            distribution_mode="easy",
            param_values=[DODGEBALL_EASIER_PARAMS_P1],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Dodgeball-Easy-E0to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenDodgeballEasyE0to3TerminalMinParamAdversarial",
    max_episode_steps=DODGEBALL_EPISODE_STEPS,
)
