import sys

from envs.registration import register as gym_register

from .procgen import ProcgenParamsEnv

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


STARPILOT_EPISODE_STEPS = 1000
STARPILOT_EASY_PARAMS_P1 = list(range(1, 500+1))
STARPILOT_EASY_PARAMS_P2 = list(range(1, 20+1))
STARPILOT_EASY_PARAMS_P3 = list(range(1, 5+1))
STARPILOT_EASY_PARAMS_P4 = list(range(1, 90+1))
STARPILOT_HARD_PARAMS_P1 = list(range(1, 500+1))
STARPILOT_HARD_PARAMS_P2 = list(range(1, 20+1))
STARPILOT_HARD_PARAMS_P3 = list(range(1, 5+1))
STARPILOT_HARD_PARAMS_P4 = list(range(1, 90+1))
STARPILOT_EASIER_PARAMS_P1 = list(range(1, 250+1))
STARPILOT_EASIER_PARAMS_P2 = list(range(1, 10+1))
STARPILOT_EASIER_PARAMS_P3 = list(range(1, 3+1))
STARPILOT_EASIER_PARAMS_P4 = list(range(1, 45+1))


class ProcgenEnvStarPilotEasy(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='easy',
            param_values=[[-1], [-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Easy-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotEasy',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenEnvStarPilotHard(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='hard',
            param_values=[[-1], [-1], [-1], [-1]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Hard-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotHard',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


# Single-parameter environments

# distribution_modes_to_use = [
#     "easy",
#     "hard",
# ]

# reward_modes_to_use = [
#     "original",
#     "terminal",
# ]

# action_spaces_to_use = [
#     "original",
# ]

# p1_params = {
#     "easy": STARPILOT_EASY_PARAMS_P1,
#     "hard": STARPILOT_HARD_PARAMS_P1,
# }

# p2_params = {
#     "easy": STARPILOT_EASY_PARAMS_P2,
#     "hard": STARPILOT_HARD_PARAMS_P2,
# }

# p3_params = {
#     "easy": STARPILOT_EASY_PARAMS_P3,
#     "hard": STARPILOT_HARD_PARAMS_P3,
# }

# p4_params = {
#     "easy": STARPILOT_EASY_PARAMS_P4,
#     "hard": STARPILOT_HARD_PARAMS_P4,
# }


##
# for mode in distribution_modes_to_use:
#     p1_params_this_mode = p1_params[mode]
#     p2_params_this_mode = p2_params[mode]
#     p3_params_this_mode = p3_params[mode]
#     p4_params_this_mode = p4_params[mode]
#     for rew in reward_modes_to_use:
#         for act in action_spaces_to_use:
#             for w in p1_params_this_mode:
#                 for t in p2_params_this_mode:
#                     for g in p3_params_this_mode:
#                         for f in p4_params_this_mode:

#                             mode_str = mode[0].upper() + mode[1:]
#                             rew_str = ""
#                             if rew == "terminal":
#                                 rew_str = "Terminal"
#                             elif rew != "original":
#                                 raise NotImplementedError
#                             if act != "original":
#                                 raise NotImplementedError

#                             class_name = f"ProcgenEnvStarPilot{mode_str}W{w}to{w}T{t}to{t}G{g}to{g}F{f}to{f}{rew_str}"
#                             print(class_name)

#                             class_definition_str = ""
#                             class_definition_str += (
#                                 f"class {class_name}(ProcgenParamsEnv):\n"
#                                 f"    def __init__(self, seed=None, **kwargs):\n"
#                                 f"        super().__init__(\n"
#                                 f"            game='starpilot',\n"
#                                 f"            distribution_mode='{mode}',\n"
#                                 f"            param_values=[[{w}], [{t}], [{g}], [{f}]],\n"
#                             )
#                             if rew == "terminal":
#                                 class_definition_str += \
#                                     f"            terminal_reward_mode=True,\n"
#                             class_definition_str += (
#                                 f"            seed=seed,\n"
#                                 f"            **kwargs,\n"
#                                 f"        )\n"
#                                 f"\n"
#                             )
#                             # print(class_definition_str)
#                             exec(class_definition_str)

#                             gym_id = ""
#                             gym_id += f"Procgen-StarPilot-{mode_str}-W{w}to{w}-T{t}to{t}-G{g}to{g}-F{f}to{f}-"
#                             if rew == "terminal":
#                                 gym_id += "Terminal-"
#                             gym_id += "v0"

#                             class_registration_str = (
#                                 f"gym_register(\n"
#                                 f"    id='{gym_id}',\n"
#                                 f"    entry_point=module_path + ':{class_name}',\n"
#                                 f"    max_episode_steps={STARPILOT_EPISODE_STEPS},\n"
#                                 f")\n"
#                                 f"\n"
#                             )
#                             # print(class_registration_str)
#                             eval(class_registration_str)


# Easy distribution mode ------------------------------------------------------
class ProcgenEnvStarPilotEasyW500to500T20to20G5to5F90to90(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='easy',
            param_values=[[500], [20], [5], [90]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Easy-W500to500-T20to20-G5to5-F90to90-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotEasyW500to500T20to20G5to5F90to90',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenEnvStarPilotEasyW500to500T20to20G5to5F90to90Terminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='easy',
            param_values=[[500], [20], [5], [90]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Easy-W500to500-T20to20-G5to5-F90to90-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotEasyW500to500T20to20G5to5F90to90Terminal',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


# Hard distribution mode ------------------------------------------------------
class ProcgenEnvStarPilotHardW500to500T20to20G5to5F90to90(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='hard',
            param_values=[[500], [20], [5], [90]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Hard-W500to500-T20to20-G5to5-F90to90-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotHardW500to500T20to20G5to5F90to90',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenEnvStarPilotHardW500to500T20to20G5to5F90to90Terminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='hard',
            param_values=[[500], [20], [5], [90]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Hard-W500to500-T20to20-G5to5-F90to90-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotHardW500to500T20to20G5to5F90to90Terminal',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


# Easier distribution mode ------------------------------------------------------
class ProcgenEnvStarPilotEasyW250to250T10to10G3to3F45to45(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='easy',
            param_values=[[250], [10], [3], [45]],
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Easy-W250to250-T10to10-G3to3-F45to45-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotEasyW250to250T10to10G3to3F45to45',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


class ProcgenEnvStarPilotEasyW250to250T10to10G3to3F45to45Terminal(ProcgenParamsEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            game='starpilot',
            distribution_mode='easy',
            param_values=[[250], [10], [3], [45]],
            terminal_reward_mode=True,
            seed=seed,
            **kwargs,
        )


gym_register(
    id='Procgen-StarPilot-Easy-W250to250-T10to10-G3to3-F45to45-Terminal-v0',
    entry_point=module_path + ':ProcgenEnvStarPilotEasyW250to250T10to10G3to3F45to45Terminal',
    max_episode_steps=STARPILOT_EPISODE_STEPS,
)


def register_starpilot_env_by_name(env_name):
    env_name_split = env_name.split('-')
    assert env_name_split[0] == "Procgen"
    assert env_name_split[1] == "StarPilot"
    easy_distrib = True if "Easy" in env_name_split else False
    hard_distrib = True if "Hard" in env_name_split else False
    terminal_mode = True if "Terminal" in env_name_split else False
    version = env_name_split[-1]
    idx_start_param = [t.startswith('W') for t in env_name_split].index(True)

    win_time_params_text = env_name_split[idx_start_param]
    win_time_params_split = win_time_params_text[1:].split('to')
    assert len(win_time_params_split) == 2
    assert win_time_params_split[0] == win_time_params_split[1]
    w = int(win_time_params_split[0])

    spawn_time_params_text = env_name_split[idx_start_param + 1]
    spawn_time_params_split = spawn_time_params_text[1:].split('to')
    assert len(spawn_time_params_split) == 2
    assert spawn_time_params_split[0] == spawn_time_params_split[1]
    t = int(spawn_time_params_split[0])

    group_params_text = env_name_split[idx_start_param + 2]
    group_params_split = group_params_text[1:].split('to')
    assert len(group_params_split) == 2
    assert group_params_split[0] == group_params_split[1]
    g = int(group_params_split[0])

    fire_time_params_text = env_name_split[idx_start_param + 3]
    fire_time_params_split = fire_time_params_text[1:].split('to')
    assert len(fire_time_params_split) == 2
    assert fire_time_params_split[0] == fire_time_params_split[1]
    f = int(fire_time_params_split[0])

    class_name = "ProcgenEnvStarPilot"
    if easy_distrib:
        class_name += "Easy"
    if hard_distrib:
        class_name += "Hard"
    class_name += f"W{w}to{w}T{t}to{t}G{g}to{g}F{f}to{f}"
    if terminal_mode:
        class_name += "Terminal"

    class_definition_str = (
        f"class {class_name}(ProcgenParamsEnv):\n"
        f"    def __init__(self, seed=None, **kwargs):\n"
        f"        super().__init__(\n"
        f"            game='starpilot',\n"
    )

    if easy_distrib:
        class_definition_str += \
        f"            distribution_mode='easy',\n"

    if hard_distrib:
        class_definition_str += \
        f"            distribution_mode='hard',\n"

    class_definition_str += \
        f"            param_values=[[{w}], [{t}], [{g}], [{f}]],\n"

    if terminal_mode:
        class_definition_str += \
        f"            terminal_reward_mode=True,\n"

    class_definition_str += (
        f"            seed=seed,\n"
        f"            **kwargs,\n"
        f"        )\n"
        f"\n"
    )
    # print(class_definition_str)
    exec(class_definition_str)

    class_registration_str = (
        f"gym_register(\n"
        f"    id='{env_name}',\n"
        f"    entry_point=module_path + ':{class_name}',\n"
        f"    max_episode_steps={STARPILOT_EPISODE_STEPS},\n"
        f")\n"
        f"\n"
    )
    # print(class_registration_str)
    eval(class_registration_str)

    setattr(sys.modules[__name__], class_name, eval(class_name))
