def env_param_config_from_name(env_name):

    # MultiRoom
    if env_name in ["multiroom_plr_1d"]:
        min_params = [1]
        max_params = [4]
        num_param_steps = [4]
        param_names = ["Mean num. rooms"]
    # Procgen
    elif env_name == "bigfish_easier_1d":
        min_params = [1]
        max_params = [10]
        num_param_steps = [10]
        param_names = ["Mean fish quota"]
    elif env_name == "bigfish_easy_1d":
        min_params = [1]
        max_params = [30]
        num_param_steps = [30]
        param_names = ["Mean fish quota"]
    elif env_name == "bigfish_hard_1d":
        min_params = [1]
        max_params = [30]
        num_param_steps = [30]
        param_names = ["Mean fish quota"]
    elif env_name == "bossfight_easy_2d":
        min_params = [1, 1]
        max_params = [9, 5]
        num_param_steps = [9, 5]
        param_names = [
            "Mean round health",
            "Mean num. rounds",
        ]
    elif env_name == "bossfight_easier_3d":
        min_params = [1, 1, 2]
        max_params = [2, 2, 3]
        num_param_steps = [2, 2, 2]
        param_names = [
            "Mean round health",
            "Mean num. rounds",
            "Mean invul. sec.",
        ]
    elif env_name == "bossfight_easy_3d":
        min_params = [1, 1, 2]
        max_params = [9, 5, 3]
        num_param_steps = [9, 5, 2]
        param_names = [
            "Mean round health",
            "Mean num. rounds",
            "Mean invul. sec.",
        ]
    elif env_name == "bossfight_hard_3d":
        min_params = [1, 1, 2]
        max_params = [9, 5, 5]
        num_param_steps = [9, 5, 4]
        param_names = [
            "Mean round health",
            "Mean num. rounds",
            "Mean invul. sec.",
        ]
    elif env_name == "caveflyer_easy_1d":
        min_params = [0]
        max_params = [3]
        num_param_steps = [4]
        param_names = [
            "Mean num. objects per chunk",
        ]
    elif env_name == "caveflyer_hard_1d":
        min_params = [0]
        max_params = [3]
        num_param_steps = [4]
        param_names = [
            "Mean num. objects per chunk",
        ]
    elif env_name == "chaser_easy_1d":
        min_params = [0]
        max_params = [3]
        num_param_steps = [4]
        param_names = [
            "Mean num. enemies",
        ]
    elif env_name == "chaser_hard_1d":
        min_params = [0]
        max_params = [3]
        num_param_steps = [4]
        param_names = [
            "Mean num. enemies",
        ]
    elif env_name == "chaser_easier_2d":
        min_params = [0, 1]
        max_params = [3, 75]
        num_param_steps = [4, 75]
        param_names = [
            "Mean num. enemies",
            "Mean orb frac.",
        ]
    elif env_name == "chaser_easy_2d":
        min_params = [0, 1]
        max_params = [3, 100]
        num_param_steps = [4, 100]
        param_names = [
            "Mean num. enemies",
            "Mean orb frac.",
        ]
    elif env_name == "chaser_hard_2d":
        min_params = [0, 1]
        max_params = [3, 100]
        num_param_steps = [4, 100]
        param_names = [
            "Mean num. enemies",
            "Mean orb frac.",
        ]
    elif env_name == "climber_easier_2d":
        min_params = [1, 0]
        max_params = [5, 20]
        num_param_steps = [5, 21]
        param_names = [
            "Mean num. platforms",
            "Mean enemy prob. (%)",
        ]
    elif env_name == "climber_easy_2d":
        min_params = [1, 0]
        max_params = [10, 20]
        num_param_steps = [10, 21]
        param_names = [
            "Mean num. platforms",
            "Mean enemy prob. (%)",
        ]
    elif env_name == "climber_hard_2d":
        min_params = [1, 0]
        max_params = [10, 50]
        num_param_steps = [10, 51]
        param_names = [
            "Mean num. platforms",
            "Mean enemy prob. (%)",
        ]
    elif env_name == "coinrun_easy_2d":
        min_params = [1, 1]
        max_params = [3, 5]
        num_param_steps = [3, 5]
        param_names = [
            "Mean difficulty",
            "Mean num. sections",
        ]
    elif env_name == "coinrun_hard_2d":
        min_params = [1, 1]
        max_params = [3, 5]
        num_param_steps = [3, 5]
        param_names = [
            "Mean difficulty",
            "Mean num. sections",
        ]
    elif env_name == "dodgeball_easier_1d":
        min_params = [0]
        max_params = [3]
        num_param_steps = [4]
        param_names = [
            "Mean num. enemies",
        ]
    elif env_name == "dodgeball_easy_1d":
        min_params = [3]
        max_params = [6]
        num_param_steps = [4]
        param_names = [
            "Mean num. enemies",
        ]
    elif env_name == "dodgeball_hard_1d":
        min_params = [3]
        max_params = [6]
        num_param_steps = [4]
        param_names = [
            "Mean num. enemies",
        ]
    elif env_name == "fruitbot_easy_3d":
        min_params = [1, 0, 0]
        max_params = [5, 60, 10]
        num_param_steps = [5, 61, 11]
        param_names = [
            "Mean num. walls",
            "Mean wall gap",
            "Mean min. bad objects",
        ]
    elif env_name == "fruitbot_hard_4d":
        min_params = [1, 0, 0, 0]
        max_params = [10, 70, 10, 5]
        num_param_steps = [10, 71, 11, 6]
        param_names = [
            "Mean num. walls",
            "Mean wall gap",
            "Mean min. bad objects",
            "Mean lock prob.",
        ]
    elif env_name == "fruitbotinv_easy_3d":
        min_params = [1, 0, 0]
        max_params = [5, 60, 10]
        num_param_steps = [5, 61, 11]
        param_names = [
            "Mean num. walls",
            "Mean wall gap (inv)",
            "Mean min. bad objects",
        ]
    elif env_name == "fruitbotinv_hard_4d":
        min_params = [1, 0, 0, 0]
        max_params = [10, 70, 10, 5]
        num_param_steps = [10, 71, 11, 6]
        param_names = [
            "Mean num. walls",
            "Mean wall gap (inv)",
            "Mean min. bad objects",
            "Mean lock prob.",
        ]
    elif env_name == "heist_easy_2d":
        min_params = [0, 0]
        max_params = [2, 3]
        num_param_steps = [3, 4]
        param_names = [
            "Mean difficulty",
            "Mean num. keys",
        ]
    elif env_name == "heist_hard_2d":
        min_params = [0, 0]
        max_params = [4, 3]
        num_param_steps = [5, 4]
        param_names = [
            "Mean difficulty",
            "Mean num. keys",
        ]
    elif env_name == "jumper_easy_1d":
        min_params = [0]
        max_params = [20]
        num_param_steps = [21]
        param_names = [
            "Mean spike prob. (%)",
        ]
    elif env_name == "jumper_hard_1d":
        min_params = [0]
        max_params = [20]
        num_param_steps = [21]
        param_names = [
            "Mean spike prob. (%)",
        ]
    elif env_name == "leaper_easy_2d":
        min_params = [0, 0]
        max_params = [3, 3]
        num_param_steps = [4, 4]
        param_names = [
            "Mean num. road lanes",
            "Mean num. water lanes",
        ]
    elif env_name == "leaper_hard_2d":
        min_params = [0, 0]
        max_params = [5, 5]
        num_param_steps = [6, 6]
        param_names = [
            "Mean num. road lanes",
            "Mean num. water lanes",
        ]
    elif env_name == "maze_easy_1d":
        min_params = [0]
        max_params = [6]
        num_param_steps = [7]
        param_names = [
            "Mean difficulty (size)"
        ]
    elif env_name == "maze_hard_1d":
        min_params = [0]
        max_params = [11]
        num_param_steps = [12]
        param_names = [
            "Mean difficulty (size)"
        ]
    elif env_name == "miner_easy_2d":
        min_params = [0, 0]
        max_params = [3, 20]
        num_param_steps = [4, 21]
        param_names = [
            "Mean num. diamonds",
            "Mean num. boulders"
        ]
    elif env_name == "miner_hard_2d":
        min_params = [0, 0]
        max_params = [12, 80]
        num_param_steps = [13, 81]
        param_names = [
            "Mean num. diamonds",
            "Mean num. boulders"
        ]
    elif env_name == "ninja_easy_2d":
        min_params = [1, 1]
        max_params = [3, 5]
        num_param_steps = [3, 5]
        param_names = [
            "Mean difficulty",
            "Mean num. sections",
        ]
    elif env_name == "ninja_hard_2d":
        min_params = [1, 1]
        max_params = [3, 5]
        num_param_steps = [3, 5]
        param_names = [
            "Mean difficulty",
            "Mean num. sections",
        ]
    elif env_name == "plunder_easier_2d":
        min_params = [1, 1]
        max_params = [8, 10]
        num_param_steps = [8, 10]
        param_names = [
            "Mean num. targets",
            "Mean juice penalty",
        ]
    elif env_name == "plunder_easy_2d":
        min_params = [1, 1]
        max_params = [20, 10]
        num_param_steps = [20, 10]
        param_names = [
            "Mean num. targets",
            "Mean juice penalty",
        ]
    elif env_name == "plunder_hard_3d":
        min_params = [1, 1, 0]
        max_params = [20, 10, 3]
        num_param_steps = [20, 10, 4]
        param_names = [
            "Mean num. targets",
            "Mean juice penalty",
            "Mean num. panels",
        ]
    elif env_name == "starpilot_easier_4d":
        min_params = [1, 1, 1, 1]
        max_params = [250, 10, 3, 45]
        num_param_steps = [250, 10, 3, 45]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots",
        ]
    elif env_name == "starpilot_easy_4d":
        min_params = [1, 1, 1, 1]
        max_params = [500, 20, 5, 90]
        num_param_steps = [500, 20, 5, 90]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots",
        ]
    elif env_name == "starpilot_hard_4d":
        min_params = [1, 1, 1, 1]
        max_params = [500, 20, 5, 90]
        num_param_steps = [500, 20, 5, 90]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots",
        ]
    elif env_name == "starpilotinv_easier_4d":
        min_params = [1, 11, 1, 46]
        max_params = [250, 20, 3, 90]
        num_param_steps = [250, 10, 3, 45]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups (inv)",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots (inv)",
        ]
    elif env_name == "starpilotinv_easy_4d":
        min_params = [1, 1, 1, 1]
        max_params = [500, 20, 5, 90]
        num_param_steps = [500, 20, 5, 90]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups (inv)",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots (inv)",
        ]
    elif env_name == "starpilotinv_hard_4d":
        min_params = [1, 1, 1, 1]
        max_params = [500, 20, 5, 90]
        num_param_steps = [500, 20, 5, 90]
        param_names = [
            "Mean finish line spawn time",
            "Mean min. time range between groups (inv)",
            "Mean max. flyer group size",
            "Mean min. time range between flyer shots (inv)",
        ]
    # BipedalWalker
    elif env_name == "bipedalwalker_full_8d":
        min_params = [0, 0, 0, 0, 0, 0, 0, 1]
        max_params = [10, 10, 10, 5, 5, 5, 5, 9]
        num_param_steps = None
        param_names = [
            'ground_roughness',
            'pit_gap_low',
            'pit_gap_high',
            'stump_height_low',
            'stump_height_high',
            'stair_height_low',
            'stair_height_high',
            'stair_steps',
        ]
    else:
        raise NotImplementedError

    return min_params, max_params, num_param_steps, param_names
