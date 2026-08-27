from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .bossfight import (
    BOSSFIGHT_EPISODE_STEPS,
    BOSSFIGHT_EASY_PARAMS_P1,
    BOSSFIGHT_EASY_PARAMS_P2,
    BOSSFIGHT_EASY_PARAMS_P3,
    BOSSFIGHT_HARD_PARAMS_P1,
    BOSSFIGHT_HARD_PARAMS_P2,
    BOSSFIGHT_HARD_PARAMS_P3,
    BOSSFIGHT_EASIER_PARAMS_P1,
    BOSSFIGHT_EASIER_PARAMS_P2,
    BOSSFIGHT_EASIER_PARAMS_P3,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# BossFight: Easy distribution mode
class ProcgenBossFightEasyH1to9R1to5I2to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASY_PARAMS_P1,
                BOSSFIGHT_EASY_PARAMS_P2,
                BOSSFIGHT_EASY_PARAMS_P3,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to9-R1to5-I2to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to9R1to5I2to3Adversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to9R1to5I2to3MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASY_PARAMS_P1,
                BOSSFIGHT_EASY_PARAMS_P2,
                BOSSFIGHT_EASY_PARAMS_P3,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to9-R1to5-I2to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to9R1to5I2to3MinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to9R1to5I2to3TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASY_PARAMS_P1,
                BOSSFIGHT_EASY_PARAMS_P2,
                BOSSFIGHT_EASY_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to9-R1to5-I2to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to9R1to5I2to3TerminalAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to9R1to5I2to3TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASY_PARAMS_P1,
                BOSSFIGHT_EASY_PARAMS_P2,
                BOSSFIGHT_EASY_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to9-R1to5-I2to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenBossFightEasyH1to9R1to5I2to3TerminalMinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


# BossFight: Hard distribution mode
class ProcgenBossFightHardH1to9R1to5I2to5Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="hard",
            param_values=[
                BOSSFIGHT_HARD_PARAMS_P1,
                BOSSFIGHT_HARD_PARAMS_P2,
                BOSSFIGHT_HARD_PARAMS_P3,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Hard-H1to9-R1to5-I2to5-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightHardH1to9R1to5I2to5Adversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightHardH1to9R1to5I2to5MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="hard",
            param_values=[
                BOSSFIGHT_HARD_PARAMS_P1,
                BOSSFIGHT_HARD_PARAMS_P2,
                BOSSFIGHT_HARD_PARAMS_P3,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Hard-H1to9-R1to5-I2to5-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightHardH1to9R1to5I2to5MinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightHardH1to9R1to5I2to5TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="hard",
            param_values=[
                BOSSFIGHT_HARD_PARAMS_P1,
                BOSSFIGHT_HARD_PARAMS_P2,
                BOSSFIGHT_HARD_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Hard-H1to9-R1to5-I2to5-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightHardH1to9R1to5I2to5TerminalAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightHardH1to9R1to5I2to5TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="hard",
            param_values=[
                BOSSFIGHT_HARD_PARAMS_P1,
                BOSSFIGHT_HARD_PARAMS_P2,
                BOSSFIGHT_HARD_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Hard-H1to9-R1to5-I2to5-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenBossFightHardH1to9R1to5I2to5TerminalMinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


# BossFight: Easier "distribution mode" - Easy distribution but with easier axes
class ProcgenBossFightEasyH1to2R1to2I2to3Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASIER_PARAMS_P1,
                BOSSFIGHT_EASIER_PARAMS_P2,
                BOSSFIGHT_EASIER_PARAMS_P3,
            ],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to2-R1to2-I2to3-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to2R1to2I2to3Adversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to2R1to2I2to3MinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASIER_PARAMS_P1,
                BOSSFIGHT_EASIER_PARAMS_P2,
                BOSSFIGHT_EASIER_PARAMS_P3,
            ],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to2-R1to2-I2to3-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to2R1to2I2to3MinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to2R1to2I2to3TerminalAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASIER_PARAMS_P1,
                BOSSFIGHT_EASIER_PARAMS_P2,
                BOSSFIGHT_EASIER_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to2-R1to2-I2to3-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenBossFightEasyH1to2R1to2I2to3TerminalAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)


class ProcgenBossFightEasyH1to2R1to2I2to3TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="bossfight",
            distribution_mode="easy",
            param_values=[
                BOSSFIGHT_EASIER_PARAMS_P1,
                BOSSFIGHT_EASIER_PARAMS_P2,
                BOSSFIGHT_EASIER_PARAMS_P3,
            ],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-BossFight-Easy-H1to2-R1to2-I2to3-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenBossFightEasyH1to2R1to2I2to3TerminalMinParamAdversarial",
    max_episode_steps=BOSSFIGHT_EPISODE_STEPS,
)
