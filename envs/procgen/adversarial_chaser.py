from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .chaser import (
    CHASER_EPISODE_STEPS,
    CHASER_EASY_PARAMS_P1,
    CHASER_EASY_PARAMS_P2,
    CHASER_HARD_PARAMS_P1,
    CHASER_HARD_PARAMS_P2,
    CHASER_EASIER_PARAMS_P1,
    CHASER_EASIER_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Chaser: Easy distribution mode
class ProcgenChaserEasyE0to3O1to100Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASY_PARAMS_P1, CHASER_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to100-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to100Adversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to100MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASY_PARAMS_P1, CHASER_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to100-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to100MinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to100TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASY_PARAMS_P1, CHASER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to100-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to100TerminalAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to100TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASY_PARAMS_P1, CHASER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to100-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to100TerminalMinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


# Chaser: Hard distribution mode
class ProcgenChaserHardE0to3O1to100Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="hard",
            param_values=[CHASER_HARD_PARAMS_P1, CHASER_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Hard-E0to3-O1to100-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserHardE0to3O1to100Adversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserHardE0to3O1to100MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="hard",
            param_values=[CHASER_HARD_PARAMS_P1, CHASER_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Hard-E0to3-O1to100-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserHardE0to3O1to100MinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserHardE0to3O1to100TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="hard",
            param_values=[CHASER_HARD_PARAMS_P1, CHASER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Hard-E0to3-O1to100-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserHardE0to3O1to100TerminalAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserHardE0to3O1to100TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="hard",
            param_values=[CHASER_HARD_PARAMS_P1, CHASER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Hard-E0to3-O1to100-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserHardE0to3O1to100TerminalMinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


# Chaser: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenChaserEasyE0to3O1to75Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASIER_PARAMS_P1, CHASER_EASIER_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to75-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to75Adversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to75MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASIER_PARAMS_P1, CHASER_EASIER_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to75-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to75MinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to75TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASIER_PARAMS_P1, CHASER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to75-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to75TerminalAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)


class ProcgenChaserEasyE0to3O1to75TerminalMinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="chaser",
            distribution_mode="easy",
            param_values=[CHASER_EASIER_PARAMS_P1, CHASER_EASIER_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Chaser-Easy-E0to3-O1to75-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenChaserEasyE0to3O1to75TerminalMinParamAdversarial",
    max_episode_steps=CHASER_EPISODE_STEPS,
)
