from copy import deepcopy
from multiprocessing.pool import ThreadPool

import numpy as np
from itertools import combinations

from eval import Evaluator
from .param_distrib import ParamDistrib
from .param_probs import (
    mean_params_from_bounds,
    normalize_param_probs,
    param_probs_from_param_mean_var_1d,
    param_probs_from_param_mean_var_mc,
    param_space_delta_bin_edges_from_param_min_max_steps,
    uniform_approximation_var,
)

from rl_utils import Reps


class Curate(ParamDistrib):

    def __init__(
            self,
            strategy,
            env_name,
            min_params,
            max_params,
            num_param_steps,
            return_threshold,
            eval_env_template,
            eval_config,
            deterministic_eval,
            params_mean_init="auto",
            params_var_init="auto",
            min_params_var_diag=None,
            reps_rounds=2,
            reps_rounds_init=4,
            samples_per_reps_round=16,
            reps_rel_entropy_bound=0.75,
            reps_min_temperature=0.05,
            coef_param_reg=None,
            coef_param_reg_init="auto",
            coef_param_reg_max_learning_penalty=0.1,
            lp_penalty="auto",
            use_lp_penalty_during_init=False,
            lp_penalty_start_params_mode="learned",
            lp_penalty_max_learning_penalty=0.5,
            params_mean_shift_on_solve="auto",
            params_var_on_update="auto",
            start_in_easiest=False,
            force_update_params_ctr=128,
            min_update_params_ctr=16,
            num_mc_samples=5000000,
            num_episodes_eval=None,
            num_processes_eval=None,
            parallel_eval_workers=None,
            eval_env_kwargs=None,
            off_curriculum_strategy=None,
            off_curriculum_strategy_prob=None,
            off_curriculum_sampling=None,
            off_curriculum_return_set=None,
            keep_solved_target_task_returns=True,
            params_mean_shift_mode="directed",
            params_mean_shift_on_solve_auto_divisor=10.,
            params_var_on_update_auto_divisor=5.,
            **kwargs,
        ):
        super().__init__(strategy, env_name, **kwargs)
        self.optimizer_resets_are_supported = True

        discrete_env_param_domains = ['MiniGrid', 'MultiGrid', 'Procgen']
        is_discrete_env_param_domains = [env_name.startswith(e) for e in discrete_env_param_domains]
        self.is_domain_discrete_env_param = any(is_discrete_env_param_domains)
        scalar_env_param_domains = ['MiniGrid', 'MultiGrid']
        self.is_domain_scalar_env_param = any([env_name.startswith(e) for e in scalar_env_param_domains])
        self.is_domain_starpilot = env_name.startswith('Procgen-StarPilot-')
        self.is_domain_starpilot_inv = env_name.startswith('Procgen-StarPilotInv-')

        # Min and max parameters
        min_params_in = [min_params] if np.isscalar(min_params) else min_params
        if self.is_domain_discrete_env_param:
            for param in min_params_in:
                assert isinstance(param, int), (
                    f"For {discrete_env_param_domains[is_discrete_env_param_domains.index(True)]} environments, "
                    f"\"min_params\" ({min_params}) must be an int or int list, but parameter {param} is a {type(param)}."
                )
        self.min_params = min_params_in
        self.num_params = len(self.min_params)

        max_params_in = [max_params] if np.isscalar(max_params) else max_params
        if self.is_domain_discrete_env_param:
            for param in max_params_in:
                assert isinstance(param, int), (
                    f"For {discrete_env_param_domains[is_discrete_env_param_domains.index(True)]} environments, "
                    f"\"max_params\" ({max_params}) must be an int or int list, but parameter {param} is a {type(param)}."
                )

        num_max_params = len(max_params_in)
        assert num_max_params == self.num_params, (
            f"Expected the dimensionality of \"min_params\" ({self.num_params}) and the dimensionality of \"max_params\" ({num_max_params}) to be the same, "
            f"but they are different."
        )
        self.max_params = max_params_in

        # Number of parameter steps in the curriculum
        num_param_steps_in = [num_param_steps] if np.isscalar(num_param_steps) else num_param_steps
        for param_step in num_param_steps_in:
            assert isinstance(param_step, int), (
                f"Argument \"num_param_steps\" ({num_param_steps}) must be a positive int or positive int list, "
                f"but step {param_step} is a {type(param_step)}."
            )
            assert param_step > 0, (
                f"Argument \"num_param_steps\" ({num_param_steps}) must be a positive int or positive int list, "
                f"but step {param_step} is not positive."
            )
        self.num_param_steps = num_param_steps_in

        if self.is_domain_discrete_env_param:
            self.param_space, self.param_delta, self.param_bin_edges = param_space_delta_bin_edges_from_param_min_max_steps(
                self.min_params,
                self.max_params,
                self.num_param_steps,
            )

        assert isinstance(reps_rounds, int), \
            f"Argument \"reps_rounds\" must be an int, but it is a {type(reps_rounds)}."

        assert isinstance(reps_rounds_init, int), \
            f"Argument \"reps_rounds_init\" must be an int, but it is a {type(reps_rounds_init)}."

        assert isinstance(samples_per_reps_round, int), \
            f"Argument \"samples_per_reps_round\" must be an int, but it is a {type(samples_per_reps_round)}."

        self.return_threshold = return_threshold
        self.eval_env_template = eval_env_template
        self.deterministic_eval = deterministic_eval
        self.reps_rounds = reps_rounds
        self.reps_rounds_init = reps_rounds_init
        self.samples_per_reps_round = samples_per_reps_round

        # Initial mean of the parameters
        self.params_mean_init = None
        if params_mean_init == "auto":
            # Initialize at the mean of each axis
            self.params_mean_init = mean_params_from_bounds(self.min_params, self.max_params)
            print(f"Automatic selection of params_mean_init:")
            print(f" -> min_params: {self.min_params}")
            print(f" -> max_params: {self.max_params}")
            print(f" -> params_mean_init: {self.params_mean_init} (min_params/2 + max_params/2)")
        else:
            # Check that the number of dimensions agrees
            params_mean_init_in = [params_mean_init] if np.isscalar(params_mean_init) else params_mean_init
            len_params_mean_init_in = len(params_mean_init_in)
            assert len_params_mean_init_in == self.num_params, (
                f"If specified, expected the dimensionality of \"params_mean_init\" ({len_params_mean_init_in}) to be the same as the dimensionality of the environment ({self.num_params}), "
                f"but it is different."
            )
            self.params_mean_init = params_mean_init_in
        self.params_mean_init = np.array(self.params_mean_init)
        self.params_mean = self.params_mean_init.copy()

        # How much to shift the parameter mean upon solve
        self.params_mean_shift_on_solve = None
        self.params_mean_shift_on_solve_auto_divisor = None
        if params_mean_shift_on_solve == "auto":
            assert params_mean_shift_on_solve_auto_divisor > 0., (
                f"Expected \"{params_mean_shift_on_solve_auto_divisor}\" to be greater than zero, but it is not."
            )
            self.params_mean_shift_on_solve_auto_divisor = params_mean_shift_on_solve_auto_divisor
            self.params_mean_shift_on_solve = [(self.max_params[p] - self.min_params[p])/self.params_mean_shift_on_solve_auto_divisor for p in range(self.num_params)]
            print(f"Automatic selection of params_mean_shift_on_solve:")
            print(f" -> self.params_mean_shift_on_solve_auto_divisor: {self.params_mean_shift_on_solve_auto_divisor}")
        else:
            # Check that the number of dimensions agrees
            params_mean_shift_on_solve_in = [params_mean_shift_on_solve] if np.isscalar(params_mean_shift_on_solve) else params_mean_shift_on_solve
            len_params_mean_shift_on_solve_in = len(params_mean_shift_on_solve_in)
            assert len_params_mean_shift_on_solve_in == self.num_params, (
                f"If specified, expected the dimensionality of \"params_mean_shift_on_solve\" ({len_params_mean_shift_on_solve_in}) to be the same as the dimensionality of the environment ({self.num_params}), "
                f"but it is different."
            )
            self.params_mean_shift_on_solve = params_mean_shift_on_solve_in
            print(f"Manual selection of params_mean_shift_on_solve:")
        print(f" -> params_mean_shift_on_solve: {self.params_mean_shift_on_solve}")

        assert params_mean_shift_mode in ["constant", "directed"]
        self.params_mean_shift_mode = params_mean_shift_mode
        print(f"Parameter mean shift on solve mode: {self.params_mean_shift_mode}")

        # Variance of the parameters to use initially
        self.params_var_init = None
        if params_var_init == "auto":
            initial_param_var_opt, _ = uniform_approximation_var(
                self.params_mean,
                self.min_params,
                self.max_params,
                self.num_param_steps,
                seed=self.seed,
            )
            print(f"Automatic selection of params_var_init:")
            print(f" -> Result of optimization: {initial_param_var_opt}")
            print(f" -> Result of optimization (diagonal var terms): {np.diag(initial_param_var_opt)}")
            self.params_var_init = initial_param_var_opt
        else:
            # Check that the number of dimensions agrees
            params_var_init_in = params_var_init
            if np.isscalar(params_var_init_in):
                params_var_init_in = [params_var_init_in]

            shape_params_var_init_in = np.array(params_var_init_in).shape
            if len(shape_params_var_init_in) == 1:
                # Convenience function to specify the diagonal
                assert shape_params_var_init_in[0] == self.num_params, (
                    f"If specified, expected the dimensionality of \"params_var_init\" ({shape_params_var_init_in[0]}) to be the same as the dimensionality of the environment ({self.num_params}) when specifying diagonal variance elements, "
                    f"but it is different."
                )
                params_var_init_in = np.diag(params_var_init_in)
            else:
                assert len(shape_params_var_init_in) == 2 and \
                    all([shape_params_var_init_in[p] == self.num_params for p in range(self.num_params)]), (
                        f"If specified, expected the shape of \"params_var_init\" ({shape_params_var_init_in}) to be compatible with the dimensionality of the environment ({(self.num_params, self.num_params)}) when setting the variance directly, "
                        f"but it is incompatible."
                )
                params_var_init_in = np.array(params_var_init_in)
            self.params_var_init = params_var_init_in
        self.params_var = self.params_var_init.copy()

        # Minimum variance for each parameter
        self.min_params_var_diag = None
        min_params_var_diag_in = min_params_var_diag
        if min_params_var_diag_in is None:
            self.min_params_var_diag = np.full(self.num_params, 1e-5)
        else:
            if np.isscalar(min_params_var_diag_in):
                min_params_var_diag_in = [min_params_var_diag_in]

            min_params_var_diag_in = np.array(min_params_var_diag_in)
            shape_min_params_var_diag_in = min_params_var_diag_in.shape

            assert len(shape_min_params_var_diag_in) == 1, (
                f"If specified, \"min_params_var_diag\" ({min_params_var_diag}) must be a vector, "
                f"but it has shape {shape_min_params_var_diag_in}."
            )

            assert shape_min_params_var_diag_in[0] == self.num_params, (
                f"If specified, expected the dimensionality of \"min_params_var_diag\" ({shape_min_params_var_diag_in[0]}) to be the same as the dimensionality of the environment ({self.num_params}), "
                f"but it is different."
            )
            self.min_params_var_diag = min_params_var_diag_in

        # Variance of the parameters to use during a curriculum update
        self.params_var_on_update = None
        self.params_var_on_update_auto_divisor = None
        if params_var_on_update == "auto":
            assert params_var_on_update_auto_divisor > 0., (
                f"Expected \"{params_var_on_update_auto_divisor}\" to be greater than zero, but it is not."
            )
            self.params_var_on_update_auto_divisor = params_var_on_update_auto_divisor
            self.params_var_on_update = np.diag([((self.max_params[p] - self.min_params[p])/self.params_var_on_update_auto_divisor)**2. for p in range(self.num_params)])
            print(f"Automatic selection of params_var_on_update:")
            print(f" -> params_var_on_update_auto_divisor: {self.params_var_on_update_auto_divisor}")
        else:
            # Check that the number of dimensions agrees
            params_var_on_update_in = params_var_on_update
            if np.isscalar(params_var_on_update_in):
                params_var_on_update_in = [params_var_on_update_in]

            shape_params_var_on_update_in = np.array(params_var_on_update_in).shape
            if len(shape_params_var_on_update_in) == 1:
                # Convenience function to specify the diagonal
                assert shape_params_var_on_update_in[0] == self.num_params, (
                    f"If specified, expected the dimensionality of \"params_var_on_update\" ({shape_params_var_on_update_in[0]}) to be the same as the dimensionality of the environment ({self.num_params}) when specifying diagonal variance elements, "
                    f"but it is different."
                )
                params_var_on_update_in = np.diag(params_var_on_update_in)
            else:
                assert len(shape_params_var_on_update_in) == 2 and \
                    all([shape_params_var_on_update_in[p] == self.num_params for p in range(self.num_params)]), (
                        f"If specified, expected the shape of \"params_var_on_update\" ({shape_params_var_on_update_in}) to be compatible with the dimensionality of the environment ({(self.num_params, self.num_params)}) when setting the variance directly, "
                        f"but it is incompatible."
                )
                params_var_on_update_in = np.array(params_var_on_update_in)
            self.params_var_on_update = params_var_on_update_in
            print("Manual selection of params_var_on_update:")
        print(f" -> params_var_on_update: {self.params_var_on_update}")
        print(f" -> params_var_on_update (diagonal var terms): {np.diag(self.params_var_on_update)}")

        self.reps_rel_entropy_bound = reps_rel_entropy_bound
        self.reps_min_temperature = reps_min_temperature
        reps_hyperparams_to_use = {
            'rel_entropy_bound': self.reps_rel_entropy_bound,
            'min_temperature': self.reps_min_temperature,
        }
        self.reps = Reps(**reps_hyperparams_to_use)

        self.start_in_easiest = start_in_easiest

        self.coef_param_reg = None
        if coef_param_reg is not None:
            if coef_param_reg == "auto":
                # automatic determination of hyperparameter
                param_distance_norm = np.linalg.norm(np.array(self.max_params) - np.array(self.min_params))
                coef_param_reg_to_use = coef_param_reg_max_learning_penalty/param_distance_norm
                print(f"Automatic selection of coef_param_reg:")
                print(f" -> coef_param_reg_max_learning_penalty: {coef_param_reg_max_learning_penalty}")
                print(f" -> param_distance_norm: {param_distance_norm} (L2 norm of max_params ({self.max_params}) - min_params ({self.min_params}))")
                print(f" -> coef_param_reg: {coef_param_reg_to_use} (coef_param_reg_max_learning_penalty/param_distance_norm)")
            else:
                assert np.isscalar(coef_param_reg), (
                    f"If specified, expected \"coef_param_reg\" ({coef_param_reg}) to be a scalar or \"auto\", but it is neither."
                )
                coef_param_reg_to_use = coef_param_reg
            self.coef_param_reg = coef_param_reg_to_use

        self.coef_param_reg_init = None
        if coef_param_reg_init is not None:
            if coef_param_reg_init == "auto":
                # automatic determination of hyperparameter
                param_distance_norm = np.linalg.norm(np.array(self.max_params) - np.array(self.min_params))
                coef_param_reg_init_to_use = coef_param_reg_max_learning_penalty/param_distance_norm
                print(f"Automatic selection of coef_param_reg_init:")
                print(f" -> coef_param_reg_max_learning_penalty: {coef_param_reg_max_learning_penalty}")
                print(f" -> param_distance_norm: {param_distance_norm} (L2 norm of max_params ({self.max_params}) - min_params ({self.min_params}))")
                print(f" -> coef_param_reg_init: {coef_param_reg_init_to_use} (coef_param_reg_max_learning_penalty/param_distance_norm)")
            else:
                assert np.isscalar(coef_param_reg_init), (
                    f"If specified, expected \"coef_param_reg_init\" ({coef_param_reg_init}) to be a scalar or \"auto\", but it is neither."
                )
                coef_param_reg_init_to_use = coef_param_reg_init
            self.coef_param_reg_init = coef_param_reg_init_to_use

        self.lp_penalty = None
        if lp_penalty is not None:
            if lp_penalty == "auto":
                if self.num_params == 1:
                    lp_penalty_to_use = None
                    print(f"Automatic selection of lp_penalty:")
                    print(f" -> lp_penalty: {lp_penalty_to_use} (Not using lp_penalty because the curriculum space is one-dimensional.)")
                else:
                    all_combs = []
                    dims_to_check = np.arange(1, self.num_params + 1)
                    for max_dim_comb in dims_to_check:
                        this_combs = list(combinations(np.arange(self.num_params), max_dim_comb))
                        all_combs.extend(this_combs)

                    curriculum_space_corners = [np.array(self.min_params, dtype=float)]
                    for this_comb in all_combs:
                        this_corner = np.array(self.min_params, dtype=float)
                        for idx in this_comb:
                            this_corner[idx] = self.max_params[idx]
                        curriculum_space_corners.append(this_corner)
                    assert len(curriculum_space_corners) == 2**(self.num_params)

                    curriculum_vector = np.array(self.max_params, dtype=float) - np.array(self.min_params, dtype=float)
                    dist_from_shortest_curriculum_path = []
                    for params in curriculum_space_corners:
                        params_vector = params - np.array(self.min_params, dtype=float)
                        dist_along_curriculum_vector = np.dot(params_vector, curriculum_vector)/np.dot(curriculum_vector, curriculum_vector)
                        dist_from_shortest_curriculum_path.append(params_vector - dist_along_curriculum_vector*curriculum_vector)

                    dist_norm_from_shortest_curriculum_path = np.linalg.norm(dist_from_shortest_curriculum_path, axis=1)
                    max_dist_norm_from_shortest_curriculum_path = np.max(dist_norm_from_shortest_curriculum_path)
                    lp_penalty_to_use = lp_penalty_max_learning_penalty/max_dist_norm_from_shortest_curriculum_path
                    print(f"Automatic selection of lp_penalty:")
                    print(f" -> lp_penalty_max_learning_penalty: {lp_penalty_max_learning_penalty}")
                    print(f" -> max_dist_norm_from_shortest_curriculum_path: {max_dist_norm_from_shortest_curriculum_path}")
                    print(f" -> lp_penalty: {lp_penalty_to_use} (lp_penalty_max_learning_penalty/max_dist_norm_from_shortest_curriculum_path)")
            else:
                assert np.isscalar(lp_penalty), (
                    f"If specified, expected \"lp_penalty\" ({lp_penalty}) to be a scalar or \"auto\", but it is neither."
                )
                lp_penalty_to_use = lp_penalty
            self.lp_penalty = lp_penalty_to_use

        self.use_lp_penalty_during_init = use_lp_penalty_during_init

        assert lp_penalty_start_params_mode in ["min_params", "learned"]
        self.lp_penalty_start_params_mode = lp_penalty_start_params_mode
        self.lp_penalty_start_params = None
        if self.lp_penalty_start_params_mode == "min_params":
            self.lp_penalty_start_params = np.array(self.min_params, dtype=float)

        # can't use learned lp penalty start params when we haven't learned them yet after initialization
        assert not (self.lp_penalty_start_params_mode == "learned" and self.use_lp_penalty_during_init)

        self.num_envs = None
        self.env_param_matrix = None
        self.param_probs = None
        if self.is_domain_discrete_env_param:
            # Only calculate this for discrete domains
            self.num_envs = np.prod(self.num_param_steps)
            param_space_list = [p.tolist() for p in self.param_space]
            env_param_matrix_temp = np.array(np.meshgrid(*param_space_list, indexing='ij'))
            self.env_param_matrix = env_param_matrix_temp.transpose(list(range(1, self.num_params + 1)) + [0])

            uniform_prob = np.full(self.num_param_steps, 1./self.num_envs)
            if self.num_params == 1:
                uniform_prob = uniform_prob[:, np.newaxis]
            self.param_probs = normalize_param_probs(uniform_prob)

        self.last_param_update_ctr = 0
        self.last_update_mean_returns = None

        self.curriculum_update_log = None
        self.init_curriculum_update_log = None
        self.last_curriculum_update_log = None

        self.force_update_params = True if force_update_params_ctr is not None else False
        self.force_update_params_ctr = None
        if self.force_update_params:
            assert isinstance(force_update_params_ctr, int), \
                f"If specified, \"force_update_params_ctr\" must be an integer, but it is a {type(force_update_params_ctr)}."
            assert force_update_params_ctr > 0, \
                f"If specified, \"force_update_params_ctr\" must be a positive integer, but it is {force_update_params_ctr}."
            self.force_update_params_ctr = force_update_params_ctr

        self.min_update_params = True if min_update_params_ctr is not None else False
        self.min_update_params_ctr = None
        if self.min_update_params:
            assert isinstance(min_update_params_ctr, int), \
                f"If specified, \"min_update_params_ctr\" must be an integer, but it is a {type(min_update_params_ctr)}."
            assert min_update_params_ctr > 0, \
                f"If specified, \"min_update_params_ctr\" must be a positive integer, but it is {min_update_params_ctr}."
            self.min_update_params_ctr = min_update_params_ctr

        assert isinstance(num_mc_samples, int), \
            f"Argument \"num_mc_samples\" ({num_mc_samples}) must be a positive int, but it is a {type(num_mc_samples)}."
        assert num_mc_samples > 0, (
            f"Argument \"num_mc_samples\" ({num_mc_samples}) must be a positive int, "
            f"but it is not positive."
        )
        self.num_mc_samples = num_mc_samples

        # Configuration for evaluation by overriding values from target evaluation
        self.eval_config = deepcopy(eval_config)
        if num_episodes_eval is not None:
            assert isinstance(num_episodes_eval, int), \
                f"If specified, \"num_episodes_eval\" ({num_episodes_eval}) must be a positive int, but it is a {type(num_episodes_eval)}."
            assert num_episodes_eval > 0, (
                f"If specified, \"num_episodes_eval\" ({num_episodes_eval}) must be a positive int, "
                f"but it is not positive."
            )
            self.eval_config["num_episodes"] = num_episodes_eval
        if num_processes_eval is not None:
            assert isinstance(num_processes_eval, int), \
                f"If specified, \"num_processes_eval\" ({num_processes_eval}) must be a positive int, but it is a {type(num_processes_eval)}."
            assert num_processes_eval > 0, (
                f"If specified, \"num_processes_eval\" ({num_processes_eval}) must be a positive int, "
                f"but it is not positive."
            )
            self.eval_config["num_processes"] = num_processes_eval

        # Configuration for parallel evaluation
        self.parallel_evals = True if parallel_eval_workers is not None else False
        self.parallel_eval_workers = None
        if self.parallel_evals:
            assert isinstance(parallel_eval_workers, int), \
                f"If specified, argument \"parallel_eval_workers\" must be a positive int, but it is a {type(parallel_eval_workers)}."
            assert parallel_eval_workers > 0, (
                f"If specified, argument \"parallel_eval_workers\" must be a positive int, but it is not positive."
            )
            self.parallel_eval_workers = parallel_eval_workers

        self.eval_env_kwargs = {}
        if eval_env_kwargs is not None:
            self.eval_env_kwargs.update(eval_env_kwargs)

        self.eval_and_env_config = deepcopy(self.eval_config)
        self.eval_and_env_config.update(self.eval_env_kwargs)

        self.run_off_curriculum_strategy = False
        self.off_curriculum_strategy = None
        self.off_curriculum_strategy_prob = None
        self.off_curriculum_sampling = None
        self.off_curriculum_return_set = None
        if (
            off_curriculum_strategy is not None
        ):
            assert off_curriculum_strategy_prob is not None
            assert off_curriculum_sampling is not None
            assert off_curriculum_return_set is not None

            valid_strategies = ["solved", "all"]
            assert off_curriculum_strategy in valid_strategies
            self.off_curriculum_strategy = off_curriculum_strategy

            assert (
                np.isscalar(off_curriculum_strategy_prob)
                and (off_curriculum_strategy_prob >= 0.0)
                and (off_curriculum_strategy_prob <= 1.0)
            )
            self.off_curriculum_strategy_prob = off_curriculum_strategy_prob

            valid_sampling = ["deterministic", "stochastic"]
            assert off_curriculum_sampling in valid_sampling
            self.off_curriculum_sampling = off_curriculum_sampling

            valid_return_set = ["on_curriculum", "all"]
            assert off_curriculum_return_set in valid_return_set
            self.off_curriculum_return_set = off_curriculum_return_set

            self.run_off_curriculum_strategy = True

        self.keep_solved_target_task_returns = keep_solved_target_task_returns

    def _initialize(self, agent):
        if self.start_in_easiest:
            if self.is_domain_discrete_env_param:
                param_probs = np.zeros(self.num_param_steps)
                if self.num_params == 1:
                    param_probs = param_probs[:, np.newaxis]
                idx_easiest_env = np.unravel_index(0, param_probs.shape)
                param_probs[idx_easiest_env] = 1.
                self.param_probs = normalize_param_probs(param_probs)
                self.params_mean = self.env_param_matrix[idx_easiest_env]
            else:
                self.params_mean = np.array(self.min_params)
            if self.num_params == 1:
                self.params_mean = np.array([self.params_mean])
            self.params_var = np.diag(np.zeros(self.num_params))
            self.last_param_update_ctr = self.update_ctr
        else:
            if self.is_domain_discrete_env_param:
                self.param_probs = self.get_param_probs_from_param_mean_var(
                    self.params_mean_init,
                    self.params_var_init,
                )
            lp_penalty_init = self.lp_penalty if self.use_lp_penalty_during_init else None
            self._update_param_probs(
                agent,
                self.reps_rounds_init,
                starting_mean=self.params_mean_init,
                starting_var=self.params_var_init,
                coef_param_reg=self.coef_param_reg_init,
                lp_penalty=lp_penalty_init,
            )

        # Transfer last_curriculum_update_log to init_curriculum_update_log
        self.init_curriculum_update_log = deepcopy(self.last_curriculum_update_log)
        self.last_curriculum_update_log = None

        if self.lp_penalty_start_params_mode == "learned":
            self.lp_penalty_start_params = deepcopy(self.params_mean)

        # Reset optimizer if needed
        if self.enable_optimizer_resets:
            self.reset_optimizer_trigger = True

    def get_param_probs_from_param_mean_var(self, param_mean, param_var):
        param_probs = None
        use_analytic_prob_for_1d = False
        if self.num_params == 1 and use_analytic_prob_for_1d:
            param_probs = param_probs_from_param_mean_var_1d(param_mean, param_var, self.min_params, self.max_params, self.num_param_steps)
        else:
            param_probs = param_probs_from_param_mean_var_mc(param_mean, param_var, self.param_bin_edges, self.num_mc_samples)
        return param_probs

    def _sample(self, index=None):

        run_off_curriculum = False
        if (
            self.run_off_curriculum_strategy
        ):
            if self.off_curriculum_sampling == "deterministic":
                run_off_curriculum = index/self.num_processes < self.off_curriculum_strategy_prob
            elif self.off_curriculum_sampling == "stochastic":
                run_off_curriculum = self.rng.random() < self.off_curriculum_strategy_prob
            else:
                raise NotImplementedError

        if self.is_domain_discrete_env_param:
            if run_off_curriculum:
                if self.off_curriculum_strategy == "solved":
                    env_is_solved = np.full(self.param_probs.shape, False)
                    for idx in range(self.num_envs):
                        idx_unravel = np.unravel_index(idx, self.param_probs.shape)
                        this_env_params = self.env_param_matrix[idx_unravel]
                        if idx == 0 or np.all(this_env_params < self.params_mean):
                            env_is_solved[idx_unravel] = True
                    param_probs_solved = env_is_solved/env_is_solved.sum()
                    idx_1d_sampled_env = self.rng.choice(
                        list(range(self.num_envs)),
                        size=1,
                        replace=True,
                        p=param_probs_solved.flatten(),
                    )[0]

                elif self.off_curriculum_strategy == "all":
                    idx_1d_sampled_env = self.rng.choice(
                        list(range(self.num_envs)),
                        size=1,
                        replace=True,
                    )[0]
                else:
                    raise NotImplementedError

            else:
                idx_1d_sampled_env = self.rng.choice(
                    list(range(self.num_envs)),
                    size=1,
                    replace=True,
                    p=self.param_probs.flatten(),
                )[0]

            idx_sampled_env = np.unravel_index(idx_1d_sampled_env, self.param_probs.shape)
            sampled_env_params = self.env_param_matrix[idx_sampled_env]

        else:
            if run_off_curriculum:
                if self.off_curriculum_strategy == "solved":
                    sampled_env_params = []

                    for p in range(self.num_params):
                        this_param_min = self.min_params[p]
                        this_param_mean = self.params_mean[p]
                        this_param_sampled = None

                        if this_param_min == this_param_mean:
                            this_param_sampled = this_param_mean
                        else:
                            this_param_sampled = self.rng.uniform(
                                low=this_param_min,
                                high=this_param_mean,
                            )
                        sampled_env_params.append(this_param_sampled)

                    sampled_env_params = np.array(sampled_env_params)

                elif self.off_curriculum_strategy == "all":
                    sampled_env_params = np.array([
                        self.rng.uniform(low=self.min_params[p], high=self.max_params[p]) for p in range(self.num_params)
                    ])

                else:
                    raise NotImplementedError

            else:
                sampled_env_params = self.rng.multivariate_normal(
                    self.params_mean,
                    self.params_var,
                )

            if self.num_params == 1:
                sampled_env_params = sampled_env_params[0]

        if self.num_params == 1:
            sampled_env_params = np.array([sampled_env_params])

        if self.is_domain_discrete_env_param:
            sampled_param = [int(p) for p in sampled_env_params]
        else:
            sampled_param = sampled_env_params.tolist()

        # Return a scalar if the parameter is only 1 dimensional
        if self.is_domain_scalar_env_param:
            assert self.num_params == 1
            sampled_param = sampled_param[0]

        return sampled_param

    def _update(self, returns=None, agent=None):
        returns_flattened = np.concatenate(returns)
        self.last_update_mean_returns = np.mean(returns_flattened)
        self.last_update_mean_returns_on_curriculum = self.last_update_mean_returns
        if self.run_off_curriculum_strategy:
            if self.off_curriculum_return_set == "on_curriculum":
                assert self.off_curriculum_sampling == "deterministic"
                on_curriculum_returns = [
                    r for idx, r in enumerate(returns)
                    if idx/self.num_processes >= self.off_curriculum_strategy_prob
                ]
                self.last_update_mean_returns_on_curriculum = np.mean(np.concatenate(on_curriculum_returns))
            elif self.off_curriculum_return_set != "all":
                raise NotImplementedError

        # Reset curriculum_update_log
        self.curriculum_update_log = None

        update_from_solve = self.last_update_mean_returns_on_curriculum >= self.return_threshold
        update_from_ctr = self.force_update_params and ((self.update_ctr - self.last_param_update_ctr) >= self.force_update_params_ctr)
        if update_from_solve or update_from_ctr:
            if (not self.min_update_params) or ((self.update_ctr - self.last_param_update_ctr) >= self.min_update_params_ctr):
                # Trigger update
                if update_from_solve:
                    if self.num_params == 1 or self.params_mean_shift_mode == "constant":
                        update_starting_mean  = self.params_mean + self.params_mean_shift_on_solve

                    elif self.params_mean_shift_mode == "directed":
                        directed_pmsos_vec = None

                        vec_to_target = np.array(self.max_params, float) - self.params_mean
                        vec_to_target_norm = np.linalg.norm(vec_to_target)

                        pmsos_vec = np.array(self.params_mean_shift_on_solve, float)
                        pmsos_vec_norm = np.linalg.norm(pmsos_vec)

                        if vec_to_target_norm >= 1.e-5 and pmsos_vec_norm >= 1.e-5:
                            dir_vec_to_target = vec_to_target / vec_to_target_norm
                            dir_pmsos_vec = pmsos_vec / pmsos_vec_norm

                            halfway_vec = dir_pmsos_vec - dir_vec_to_target
                            halfway_vec_norm = np.linalg.norm(halfway_vec)

                            householder_mat = None
                            if halfway_vec_norm >= 1.e-5:
                                dir_halfway_vec = halfway_vec / halfway_vec_norm
                                householder_mat = np.eye(self.num_params) - 2*dir_halfway_vec[:,np.newaxis] * dir_halfway_vec[:,np.newaxis].T
                            else:
                                householder_mat = np.eye(self.num_params)
                            directed_pmsos_vec = householder_mat @ pmsos_vec

                        else:
                            directed_pmsos_vec = pmsos_vec

                        update_starting_mean = self.params_mean + directed_pmsos_vec

                    else:
                        raise NotImplementedError
                else:
                    update_starting_mean = self.params_mean
                self._update_param_probs(
                    agent,
                    self.reps_rounds,
                    starting_mean=update_starting_mean,
                    starting_var=self.params_var_on_update,
                    coef_param_reg=self.coef_param_reg,
                    lp_penalty=self.lp_penalty,
                )
                self.curriculum_update_log = self.last_curriculum_update_log

        if self.update_ctr == 1:
            # On the first update, we output two logs, one for initialization and the second if a curriculum update was run
            self.curriculum_update_log = [
                self.init_curriculum_update_log,
                self.last_curriculum_update_log,
            ]

        return True

    def _get_stats(self):
        params_mean_for_stats = self.params_mean[0] if self.num_params == 1 else self.params_mean.tolist()
        params_var_for_stats = self.params_var[[0][0]] if self.num_params == 1 else self.params_var.tolist()
        params_var_diag_for_stats = self.params_var[[0][0]] if self.num_params == 1 else np.diag(self.params_var).tolist()

        stats = dict(
            params_mean=params_mean_for_stats,
            params_var=params_var_for_stats,
            params_var_diag=params_var_diag_for_stats,
            last_param_update_ctr=self.last_param_update_ctr,
            last_update_mean_returns=self.last_update_mean_returns,
            last_update_on_curriculum_mean_returns=self.last_update_mean_returns_on_curriculum,
            curriculum_update_log=self.curriculum_update_log,
        )
        return stats

    def _eval_agent_on_env(self, agent, eval_env_name):
        evaluator = Evaluator(
            [eval_env_name],
            **self.eval_and_env_config,
        )
        eval_stats = evaluator.evaluate(
            agent,
            deterministic=self.deterministic_eval,
            accumulator='mean',
            return_test_records=True,
        )
        eval_return = eval_stats[f"test_returns:{eval_env_name}"]
        episode_lengths = eval_stats[f"test_records:{eval_env_name}"]["episode_lengths"]
        total_eval_steps = np.sum(episode_lengths)
        return eval_return, total_eval_steps

    def _eval_agent_on_env_param_args(self, agent, eval_env_name, params):
        eval_env_and_param_config = deepcopy(self.eval_and_env_config)
        eval_env_and_param_config['env_params'] = params
        evaluator = Evaluator(
            [eval_env_name],
            **eval_env_and_param_config,
        )
        eval_stats = evaluator.evaluate(
            agent,
            deterministic=self.deterministic_eval,
            accumulator='mean',
            return_test_records=True,
        )
        eval_return = eval_stats[f"test_returns:{eval_env_name}"]
        episode_lengths = eval_stats[f"test_records:{eval_env_name}"]["episode_lengths"]
        total_eval_steps = np.sum(episode_lengths)
        return eval_return, total_eval_steps

    def _update_param_probs(self, agent, num_reps_rounds, starting_mean=None, starting_var=None, coef_param_reg=None, lp_penalty=None):
        round_params_mean = starting_mean.copy() if starting_mean is not None else self.params_mean.copy()
        round_params_var = starting_var.copy() if starting_var is not None else self.params_var.copy()

        rounds_log = []

        for round in range(num_reps_rounds):

            round_params_mean_init = round_params_mean.copy()
            round_params_var_init = round_params_var.copy()

            params_for_reps_buffer = []
            rewards_for_reps_buffer = []
            param_idxs_for_reps_buffer = []
            total_eval_steps_buffer = []

            if self.parallel_evals:
                # Parallel evaluation

                # Sample parameters
                all_step_params = self.rng.multivariate_normal(
                    round_params_mean,
                    round_params_var,
                    size=self.samples_per_reps_round,
                )

                num_eval_calls = np.ceil(self.samples_per_reps_round/self.parallel_eval_workers).astype(int)

                for idx_eval in range(num_eval_calls):

                    idx_start = idx_eval*self.parallel_eval_workers
                    idx_samples_this_eval = range(idx_start, np.min((idx_start + self.parallel_eval_workers, self.samples_per_reps_round)))
                    num_samples_this_eval = len(idx_samples_this_eval)
                    params_to_eval = all_step_params[idx_samples_this_eval,:]

                    env_names_to_eval = []
                    env_idx_params = []
                    if self.is_domain_discrete_env_param:
                        for p in params_to_eval:
                            this_env_name_to_eval, this_idx_params = self._env_name_from_params(p)
                            env_names_to_eval.append(this_env_name_to_eval)
                            env_idx_params.append(this_idx_params)

                            if self.is_domain_starpilot:
                                from envs.registration import spec as gym_spec
                                from envs.procgen.starpilot import register_starpilot_env_by_name
                                from gym import error as gym_error

                                try:
                                    env_spec = gym_spec(this_env_name_to_eval)
                                except gym_error.UnregisteredEnv:
                                    # Create env on an as-needed basis
                                    register_starpilot_env_by_name(this_env_name_to_eval)
                                    env_spec = gym_spec(this_env_name_to_eval)

                            if self.is_domain_starpilot_inv:
                                from envs.registration import spec as gym_spec
                                from envs.procgen.starpilot_inv import register_starpilot_inv_env_by_name
                                from gym import error as gym_error

                                try:
                                    env_spec = gym_spec(this_env_name_to_eval)
                                except gym_error.UnregisteredEnv:
                                    # Create env on an as-needed basis
                                    register_starpilot_inv_env_by_name(this_env_name_to_eval)
                                    env_spec = gym_spec(this_env_name_to_eval)
                    else:
                        env_names_to_eval = [self.eval_env_template for _ in range(num_samples_this_eval)]

                    num_workers_this_eval = len(env_names_to_eval)
                    worker_pool = ThreadPool(num_workers_this_eval)
                    if self.is_domain_discrete_env_param:
                        pool_arguments = [(agent, this_env_name_to_eval) for this_env_name_to_eval in env_names_to_eval]
                        pool_results = worker_pool.starmap(self._eval_agent_on_env, pool_arguments)
                        eval_return_pool, total_eval_steps_pool = zip(*pool_results)
                    else:
                        pool_arguments = [(agent, env_names_to_eval[w], params_to_eval[w]) for w in range(num_samples_this_eval)]
                        pool_results = worker_pool.starmap(self._eval_agent_on_env_param_args, pool_arguments)
                        eval_return_pool, total_eval_steps_pool = zip(*pool_results)

                    for this_params_to_eval in params_to_eval:
                        params_for_reps_buffer.append(this_params_to_eval.tolist())
                    rewards_for_reps_buffer.extend(eval_return_pool)
                    param_idxs_for_reps_buffer.extend(env_idx_params)
                    total_eval_steps_buffer.extend(total_eval_steps_pool)

            else:
                # Series evaluation
                for step in range(self.samples_per_reps_round):

                    # Sample parameters
                    step_params = self.rng.multivariate_normal(
                        round_params_mean,
                        round_params_var,
                    )

                    # Determine which env to run
                    if self.is_domain_discrete_env_param:
                        eval_env_name, idx_param = self._env_name_from_params(step_params)

                        if self.is_domain_starpilot:
                            from envs.registration import spec as gym_spec
                            from envs.procgen.starpilot import register_starpilot_env_by_name
                            from gym import error as gym_error

                            try:
                                env_spec = gym_spec(eval_env_name)
                            except gym_error.UnregisteredEnv:
                                # Create env on an as-needed basis
                                register_starpilot_env_by_name(eval_env_name)
                                env_spec = gym_spec(eval_env_name)

                        eval_return, total_eval_steps = self._eval_agent_on_env(agent, eval_env_name)

                    else:
                        eval_return, total_eval_steps = self._eval_agent_on_env_param_args(agent, self.eval_env_template, step_params)
                        idx_param = None

                    params_for_reps_buffer.append(step_params.tolist())
                    rewards_for_reps_buffer.append(eval_return)
                    param_idxs_for_reps_buffer.append(idx_param)
                    total_eval_steps_buffer.append(total_eval_steps)

            learning_potential = np.array(rewards_for_reps_buffer)/self.return_threshold

            # Tasks that are solved receive 0 learning progress
            x_solved = None
            if self.keep_solved_target_task_returns:
                is_not_target = None
                if self.is_domain_discrete_env_param:
                    idx_param_target = [len(self.param_space[p]) - 1 for p in range(self.num_params)]
                    is_not_target = [idx != idx_param_target for idx in param_idxs_for_reps_buffer]
                else:
                    is_not_target = [np.any(np.array(this_params) < np.array(self.max_params)) for this_params in params_for_reps_buffer]
                x_solved = np.where(np.logical_and(learning_potential >= 1., is_not_target))[0]
            else:
                x_solved = np.where(learning_potential >= 1.)[0]
            learning_potential[x_solved] = 0.

            # Apply regularization to learning potential, which serves to penalize tasks by difficulty as a proxy
            if coef_param_reg is not None:
                learning_potential -= coef_param_reg*np.linalg.norm(np.array(params_for_reps_buffer) - np.array(self.min_params), axis=1)

            # Apply learning potential penalty
            if lp_penalty is not None:
                learning_potential_penalties = np.zeros(self.samples_per_reps_round)
                curriculum_vector = np.array(self.max_params, float) - self.lp_penalty_start_params
                for idx_p, p in enumerate(params_for_reps_buffer):
                    params_vector = p - self.lp_penalty_start_params
                    dist_along_curriculum_vector = np.dot(params_vector, curriculum_vector)/np.dot(curriculum_vector, curriculum_vector)
                    dist_to_curriculum_vector = (params_vector - dist_along_curriculum_vector*curriculum_vector)
                    learning_potential_penalties[idx_p] = lp_penalty*np.linalg.norm(dist_to_curriculum_vector)
                learning_potential -= learning_potential_penalties

            new_params_mean, new_params_var, round_reps_info = self.reps.policy_from_samples_and_rewards(
                params_for_reps_buffer,
                learning_potential,
            )

            # clip parameters
            for p in range(self.num_params):
                round_params_mean[p] = np.clip(new_params_mean[p], self.min_params[p], self.max_params[p])

            # Rescale covariance matrix to meet minimum variance along a particular dimension
            new_params_var_diag = np.diag(new_params_var)
            if any([new_params_var_diag[p] < self.min_params_var_diag[p] for p in range(self.num_params)]):
                marg_std_mat = np.diag(np.sqrt(new_params_var_diag))
                marg_std_mat_inv = np.linalg.inv(marg_std_mat)
                correlation_matrix = marg_std_mat_inv @ new_params_var @ marg_std_mat_inv

                new_params_var_diag_above_min = np.array([np.max([new_params_var_diag[p], self.min_params_var_diag[p]]) for p in range(self.num_params)])
                new_marg_std_mat = np.diag(np.sqrt(new_params_var_diag_above_min))
                new_params_var = new_marg_std_mat @ correlation_matrix @ new_marg_std_mat

            round_params_var = new_params_var

            this_round_log = {
                "param_mean_init": round_params_mean_init.tolist(),
                "param_var_init": round_params_var_init.tolist(),
                "param_samples": params_for_reps_buffer,
                "reward_samples": rewards_for_reps_buffer,
                "lp_samples": learning_potential.tolist(),
                "params_mean_after_reps": new_params_mean.tolist(),
                "params_var_after_reps": new_params_var.tolist(),
                "params_mean": round_params_mean.tolist(),
                "params_var": round_params_var.tolist(),
                "total_eval_steps": total_eval_steps_buffer,
            }
            rounds_log.append(this_round_log)

        self.params_mean = round_params_mean.copy()
        self.params_var = round_params_var.copy()
        if self.is_domain_discrete_env_param:
            self.param_probs = self.get_param_probs_from_param_mean_var(
                self.params_mean,
                self.params_var,
            )
        self.last_param_update_ctr = self.update_ctr

        self.last_curriculum_update_log = rounds_log

        # Reset optimizer if needed
        if self.enable_optimizer_resets:
            self.reset_optimizer_trigger = True

    def _env_name_from_params(self, params):
        idx_param = [np.argmin(np.abs(self.param_space[p] - params[p])) for p in range(self.num_params)]
        eval_env_name = self.eval_env_template

        # Iterate through parameters in reverse order, so e.g., %P10 would get replaced before %P1
        for p in range(self.num_params - 1, -1, -1):
            assert self.is_domain_discrete_env_param
            this_env_param = int(self.param_space[p][idx_param[p]])
            param_token = f"%P{p+1}"
            assert param_token in eval_env_name, (
                f"Expected to find token \"{param_token}\" in the eval env name ({eval_env_name}), but it doesn't exist."
            )
            eval_env_name = eval_env_name.replace(param_token, str(this_env_param))

        return eval_env_name, idx_param
