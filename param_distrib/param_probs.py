from typing import List

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


def param_space_delta_bin_edges_from_param_min_max_steps(
    min_params: List, max_params: List, num_param_steps: List[int]
):

    num_params = len(min_params)
    assert len(max_params) == num_params
    assert len(num_param_steps) == num_params

    param_space = [
        np.linspace(min_params[p], max_params[p], num_param_steps[p])
        for p in range(num_params)
    ]

    param_delta = [
        (max_params[p] - min_params[p]) / (num_param_steps[p] - 1.0)
        for p in range(num_params)
    ]

    param_bin_edges = []
    for p in range(num_params):
        param_bin_edges_temp = np.linspace(
            min_params[p] - param_delta[p] / 2.0,
            max_params[p] + param_delta[p] / 2.0,
            num_param_steps[p] + 1,
        )
        param_bin_edges_temp[0] = -np.inf
        param_bin_edges_temp[-1] = np.inf
        param_bin_edges.append(param_bin_edges_temp)

    return param_space, param_delta, param_bin_edges


def param_probs_from_param_mean_var_1d(
    param_mean, param_var, min_params, max_params, num_param_steps
):

    param_space, param_delta, _ = param_space_delta_bin_edges_from_param_min_max_steps(
        min_params,
        max_params,
        num_param_steps,
    )

    assert len(param_mean) == 1
    param_mean_1d = param_mean[0]
    assert np.isscalar(param_mean_1d)
    param_var_1d = param_var[0, 0]
    assert np.isscalar(param_var_1d)
    param_delta_1d = param_delta[0]
    assert np.isscalar(param_delta_1d)
    min_params_1d = min_params[0]
    assert np.isscalar(min_params_1d)
    max_params_1d = max_params[0]
    assert np.isscalar(max_params_1d)
    param_space_1d = param_space[0]

    param_std_1d = np.sqrt(param_var_1d)
    probs_middle_params = stats.norm.cdf(
        param_space_1d[1:-1] + 0.5 * param_delta_1d, param_mean_1d, param_std_1d
    ) - stats.norm.cdf(
        param_space_1d[0:-2] + 0.5 * param_delta_1d, param_mean_1d, param_std_1d
    )
    param_probs = [
        stats.norm.cdf(
            min_params_1d + 0.5 * param_delta_1d, param_mean_1d, param_std_1d
        ),
        *probs_middle_params.tolist(),
        1.0
        - stats.norm.cdf(
            max_params_1d - 0.5 * param_delta_1d, param_mean_1d, param_std_1d
        ),
    ]
    param_probs = np.array(param_probs)[:, np.newaxis]

    return normalize_param_probs(param_probs)


def param_probs_from_param_mean_var_mc(
    param_mean: np.ndarray,
    param_var: np.ndarray,
    param_bin_edges: List,
    num_mc_samples: int = 100000,
):

    num_params = len(param_mean)
    assert param_var.shape == (num_params, num_params)
    assert len(param_bin_edges) == num_params

    # Sample from the distribution
    sampled_params = np.random.multivariate_normal(
        param_mean,
        param_var,
        size=num_mc_samples,
    )

    # Bin the samples
    param_histogram, _ = np.histogramdd(sampled_params, bins=param_bin_edges)
    if num_params == 1:
        param_histogram = param_histogram[:, np.newaxis]

    # Add 1 to each cell to ensure there is a non-zero probability everywhere
    param_histogram_adj = param_histogram + 1

    return normalize_param_probs(param_histogram_adj)


def normalize_param_probs(weights: np.ndarray):
    total_weights = np.sum(weights)
    try:
        normalized_weights = weights / total_weights if total_weights > 1.0e-8 else weights
    except FloatingPointError as e:
        with np.errstate(under='warn'):
            normalized_weights = weights / total_weights if total_weights > 1.0e-8 else weights
    except Exception as e:
        raise(e)
    return normalized_weights


def mean_params_from_bounds(min_params, max_params):
    num_params = len(min_params)
    assert len(max_params) == num_params
    mean_params_from_bounds = [
        (min_params[p] + max_params[p]) / 2.0 for p in range(num_params)
    ]
    return mean_params_from_bounds


def uniform_approximation_var(
    params_mean,
    min_params,
    max_params,
    num_param_steps,
    method="Nelder-Mead",
    regularize=True,
    num_opt_tries=100,
    seed=None,
):

    num_params = len(params_mean)

    params_var_diag_opt = []
    all_opt_successes = []
    all_opt_xs = []
    all_opt_errors = []
    all_opt_errors_no_reg = []

    def prob_error(x, regularize=regularize):
        if np.isscalar(x):
            x = [x]
        probs = param_probs_from_param_mean_var_1d(
            [params_mean[p]],
            np.diag(x),
            [min_params[p]],
            [max_params[p]],
            [num_param_steps[p]],
        )
        error = probs.max() - probs.min()
        if regularize:
            error += 1e-5 * x[0]
        return error

    rng = np.random.default_rng(seed=seed)

    for p in range(num_params):

        opt_successes = []
        opt_xs = []
        opt_errors = []
        opt_errors_no_reg = None
        cluster_labels = None

        if num_param_steps[p] >= 3:
            x_inits = rng.uniform(0.5, 1000.0, size=num_opt_tries)

            for x_init in x_inits:
                opt_output = minimize(
                    prob_error, x_init, method=method, bounds=((1e-5, None),)
                )
                opt_successes.append(opt_output.success)
                opt_xs.append(opt_output.x[0])
                opt_errors.append(opt_output.fun)

            opt_errors_no_reg = [prob_error(x, regularize=False) for x in opt_xs]

            num_successes = np.count_nonzero(opt_successes)
            assert num_successes >= 80
            opt_xs_candidates = np.array(opt_xs)[np.array(opt_successes)]
            opt_errors_candidates = np.array(opt_errors)[np.array(opt_successes)]
            from sklearn.cluster import DBSCAN

            clusters = DBSCAN(
                eps=0.5,
                min_samples=50,
                metric="euclidean",
            ).fit(
                opt_xs_candidates.reshape(-1, 1), opt_errors_candidates.reshape(-1, 1)
            )
            cluster_labels = clusters.labels_
            num_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
            assert num_clusters > 0
            mean_cluster_errors = [
                np.mean(opt_errors_candidates[cluster_labels == c])
                for c in range(num_clusters)
            ]
            idx_cluster_to_use = np.argmin(mean_cluster_errors)
            params_var_diag_to_use = np.mean(
                opt_xs_candidates[cluster_labels == idx_cluster_to_use]
            )

        else:
            # At least 3 steps are required to use optimization
            # So just set to 1
            params_var_diag_to_use = 1.0
            opt_successes = None
            opt_xs = None
            opt_errors = None
            opt_errors_no_reg = None
            cluster_labels = None

        params_var_diag_opt.append(params_var_diag_to_use)
        all_opt_successes.append(opt_successes)
        all_opt_xs.append(opt_xs)
        all_opt_errors.append(opt_errors)
        all_opt_errors_no_reg.append(opt_errors_no_reg)

    params_var_opt = np.diag(params_var_diag_opt)

    info = {
        "successes": all_opt_successes,
        "xs": all_opt_xs,
        "errors": all_opt_errors,
        "errors_no_reg": all_opt_errors_no_reg,
        "cluster_labels": cluster_labels,
    }

    return params_var_opt, info
