from envs.registration import register as gym_register

from .adversarial_procgen import AdversarialProcgenParamsEnv

from .miner import (
    MINER_EPISODE_STEPS,
    MINER_EASY_PARAMS_P1,
    MINER_EASY_PARAMS_P2,
    MINER_HARD_PARAMS_P1,
    MINER_HARD_PARAMS_P2,
)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


# Miner: Easy distribution mode
class ProcgenMinerEasyD0to3B0to20Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="easy",
            param_values=[MINER_EASY_PARAMS_P1, MINER_EASY_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Easy-D0to3-B0to20-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerEasyD0to3B0to20Adversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerEasyD0to3B0to20MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="easy",
            param_values=[MINER_EASY_PARAMS_P1, MINER_EASY_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Easy-D0to3-B0to20-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerEasyD0to3B0to20MinParamAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerEasyD0to3B0to20TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="easy",
            param_values=[MINER_EASY_PARAMS_P1, MINER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Easy-D0to3-B0to20-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerEasyD0to3B0to20TerminalAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerEasyD0to3B0to20TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="easy",
            param_values=[MINER_EASY_PARAMS_P1, MINER_EASY_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Easy-D0to3-B0to20-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerEasyD0to3B0to20TerminalMinParamAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


# Miner: Hard distribution mode
class ProcgenMinerHardD0to12B0to80Adversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="hard",
            param_values=[MINER_HARD_PARAMS_P1, MINER_HARD_PARAMS_P2],
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Hard-D0to12-B0to80-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerHardD0to12B0to80Adversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerHardD0to12B0to80MinParamAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="hard",
            param_values=[MINER_HARD_PARAMS_P1, MINER_HARD_PARAMS_P2],
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Hard-D0to12-B0to80-MinParam-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerHardD0to12B0to80MinParamAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerHardD0to12B0to80TerminalAdversarial(AdversarialProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="hard",
            param_values=[MINER_HARD_PARAMS_P1, MINER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Hard-D0to12-B0to80-Terminal-Adversarial-v0",
    entry_point=module_path + ":ProcgenMinerHardD0to12B0to80TerminalAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)


class ProcgenMinerHardD0to12B0to80TerminalMinParamAdversarial(
    AdversarialProcgenParamsEnv
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game="miner",
            distribution_mode="hard",
            param_values=[MINER_HARD_PARAMS_P1, MINER_HARD_PARAMS_P2],
            terminal_reward_mode=True,
            min_param_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id="Procgen-Miner-Hard-D0to12-B0to80-Terminal-MinParam-Adversarial-v0",
    entry_point=module_path
    + ":ProcgenMinerHardD0to12B0to80TerminalMinParamAdversarial",
    max_episode_steps=MINER_EPISODE_STEPS,
)
