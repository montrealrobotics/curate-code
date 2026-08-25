# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import re
import time
import timeit
import logging
from arguments import parser

import torch
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
from baselines.logger import HumanOutputFormat

display = None

# if sys.platform.startswith('linux'):
#     print('Setting up virtual display')

#     import pyvirtualdisplay
#     display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
#     display.start()

from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.minigrid import *
from envs.runners.adversarial_runner import AdversarialRunner 
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images, make_param_distrib_args
from eval import Evaluator


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()
    
    # === Reading random seed from an environment variable and other random seed processing ===
    seed_from_env_variable = None
    if args.seed == -99:
        assert args.seed_variable is not None, \
            f"Expected seed_variable ({args.seed_variable}) to be specified when using seed == -99, but it is not defined."
        # examine seed variable to see whether we need to do some calculations
        if any([m in args.seed_variable for m in ['+','-','*','/']]):
            seed_env_variables = re.split(r'[\+\-\*\/]', args.seed_variable)
            seed_variable_str = deepcopy(args.seed_variable)
            for var in seed_env_variables:
                assert var in os.environ, \
                    f"Expected variable \"{var}\" to be defined as an environment variable when using seed == -99 and math characters, but it is missing."
                var_value = os.environ[var]
                assert var_value.isdigit(), \
                    f"Expected variable \"{var}\" to be a digit, but it is not."
                seed_variable_str = seed_variable_str.replace(var, var_value)
            seed_from_env_variable = eval(seed_variable_str)
        else:
            assert args.seed_variable in os.environ, \
                f"Expected variable \"{args.seed_variable}\" to be defined as an environment variable when using seed == -99 and no math characters, but it is missing."
            seed_from_env_variable = int(os.environ[args.seed_variable])
        args.seed = seed_from_env_variable if args.seed_variable_offset is None else seed_from_env_variable + args.seed_variable_offset
    if args.append_seed_to_xpid:
        args.xpid = args.xpid + f"-s{args.seed}"
    if args.test_seed == -99:
        assert args.seed_variable is not None, \
            f"Expected seed_variable ({args.seed_variable}) to be specified when using test_seed == -99, but it is not defined."
        args.test_seed = seed_from_env_variable
    if args.test_offset_seed:
        # This can support up to 1000 random seeds (0-999) and up to 10000 test_num_processes (0-9999) before there are seed overlaps
        # The +10000000 offsets from the train seeds
        args.test_seed = args.test_seed*10000 + 10000000

    # === Configure logging ==
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    filewriter = FileWriter(
        xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir, assert_no_prior_xpid=args.assert_no_prior_xpid
    )
    screenshot_dir = os.path.join(log_dir, args.xpid, 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir, exist_ok=True)

    def log_stats(stats):
        filewriter.log(stats)
        if args.verbose:
            HumanOutputFormat(sys.stdout).writekvs(stats)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    # === Determine device ====
    assert not (args.no_cuda and args.assert_cuda), \
        f"Incompatible arguments: no_cuda ({args.no_cuda}) and assert_cuda ({args.assert_cuda}) can't both be True."
    is_cuda_available = torch.cuda.is_available()
    if args.assert_cuda:
        assert is_cuda_available, \
            f"Expected CUDA to be available when assert_cuda ({args.assert_cuda}) is True, but it is not."
    args.cuda = not args.no_cuda and is_cuda_available
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        torch.backends.cudnn.benchmark = True
        print('Using CUDA\n')

    # === Create parallel envs ===
    venv, ued_venv = create_parallel_env(args)

    is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
    is_paired = args.ued_algo in ['paired', 'flexible_paired']
    is_param_distrib = args.ued_algo == 'parameter_distribution'

    agent = make_agent(name='agent', env=venv, args=args, device=device)
    adversary_agent, adversary_env = None, None
    if is_paired or args.use_accel_paired:
        adversary_agent = make_agent(name='adversary_agent', env=venv, args=args, device=device)

    if is_training_env:
        adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
    if args.ued_algo == 'domain_randomization' and args.use_plr and not args.use_reset_random_dr:
        adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
        adversary_env.random()

    # === Create runner ===
    plr_args = None
    if args.use_plr:
        plr_args = make_plr_args(args, venv.observation_space, venv.action_space)
    param_distrib_args = None
    if is_param_distrib:
        param_distrib_args = make_param_distrib_args(args, device)

    train_runner = AdversarialRunner(
        args=args,
        venv=venv,
        agent=agent, 
        ued_venv=ued_venv, 
        adversary_agent=adversary_agent,
        adversary_env=adversary_env,
        flexible_protagonist=False,
        train=True,
        plr_args=plr_args,
        param_distrib_args=param_distrib_args,
        device=device)

    # === Configure checkpointing ===
    timer = timeit.default_timer
    initial_update_count = 0
    last_logged_update_at_restart = -1
    checkpoint_path = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    )
    ## This is only used for the first iteration of finetuning
    if args.xpid_finetune:
        model_fname = f'{args.model_finetune}.tar'
        base_checkpoint_path = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid_finetune, model_fname))
        )

    def checkpoint(checkpoint_path, index=None):
        if args.disable_checkpoint:
            return
        safe_checkpoint({'runner_state_dict': train_runner.state_dict()}, 
                        checkpoint_path,
                        index=index, 
                        archive_interval=args.archive_interval)
        logging.info("Saved checkpoint to %s", checkpoint_path)


    # === Load checkpoint ===
    if args.checkpoint and os.path.exists(checkpoint_path):
        checkpoint_states = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        last_logged_update_at_restart = filewriter.latest_tick() # ticks are 0-indexed updates
        train_runner.load_state_dict(checkpoint_states['runner_state_dict'])
        initial_update_count = train_runner.num_updates
        logging.info(f"Resuming preempted job after {initial_update_count} updates\n") # 0-indexed next update
    elif args.xpid_finetune and not os.path.exists(checkpoint_path):
        checkpoint_states = torch.load(base_checkpoint_path)
        state_dict = checkpoint_states['runner_state_dict']
        agent_state_dict = state_dict.get('agent_state_dict')
        optimizer_state_dict = state_dict.get('optimizer_state_dict')
        train_runner.agents['agent'].algo.actor_critic.load_state_dict(agent_state_dict['agent'])
        train_runner.agents['agent'].algo.optimizer.load_state_dict(optimizer_state_dict['agent'])

    # === Set up Evaluator ===
    evaluator = None
    test_env_names = None
    num_test_envs = None
    checkpoint_best_test_model_path = None
    checkpoint_solved_test_model_path = None
    best_test_return = None
    solved_test_checkpoint_has_been_saved = None

    if args.test_env_names:
        test_env_names_input = args.test_env_names.split(',')
        test_env_names = test_env_names_input + ['aggregation'] if args.test_aggregation else test_env_names_input
        num_test_envs = len(test_env_names)

        checkpoint_best_test_model_path = [
            os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, f"model_best_test_{test_env}.tar"))
            ) for test_env in test_env_names
        ]
        checkpoint_solved_test_model_path = [
            os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, f"model_solved_test_{test_env}.tar"))
            ) for test_env in test_env_names
        ]

        best_test_return = [-np.inf for _ in range(num_test_envs)]
        solved_test_checkpoint_has_been_saved = [False for _ in range(num_test_envs)]

        eval_env_kwargs = {}
        if 'Procgen' in args.test_env_names:
            if args.env_vectorization == 'env':
                eval_env_kwargs['num_procs'] = args.test_num_processes
            eval_env_kwargs['resource_root'] = args.procgen_resource_root
            eval_env_kwargs['prebuilt_root'] = args.procgen_prebuilt_root

        evaluator = Evaluator(
            test_env_names_input,
            num_processes=args.test_num_processes, 
            num_episodes=args.test_num_episodes,
            frame_stack=args.frame_stack,
            grayscale=args.grayscale,
            num_action_repeat=args.num_action_repeat,
            use_global_critic=args.use_global_critic,
            use_global_policy=args.use_global_policy,
            device=device,
            seed=args.test_seed,
            stagger_seeds=args.test_stagger_seeds,
            env_vectorization=args.env_vectorization,
            **eval_env_kwargs)

    stop_training_due_to_test_solved = False

    # === Train ===
    last_checkpoint_idx = getattr(train_runner, args.checkpoint_basis)
    update_start_time = timer()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(initial_update_count, num_updates):
        stats = train_runner.run()

        # === Perform logging ===
        if train_runner.num_updates <= last_logged_update_at_restart:
            continue

        log = (j % args.log_interval == 0) or j == num_updates - 1
        save_screenshot = \
            args.screenshot_interval > 0 and \
                (j % args.screenshot_interval == 0)

        if log:
            # Eval
            test_stats = {}
            if evaluator is not None and (j % args.test_interval == 0 or j == num_updates - 1):
                test_stats = evaluator.evaluate(
                    train_runner.agents['agent'],
                    deterministic=args.deterministic_test_evaluation,
                    accumulator='mean',
                )
                if args.test_aggregation:
                    test_stats[f"solved_rate:aggregation"] = np.mean([test_stats[f"solved_rate:{t}"] for t in test_env_names_input])
                    test_stats[f"test_returns:aggregation"] = np.mean([test_stats[f"test_returns:{t}"] for t in test_env_names_input])
                    test_stats[f"test_episode_length:aggregation"] = np.mean([test_stats[f"test_episode_length:{t}"] for t in test_env_names_input])
                stats.update(test_stats)

                for idx_test in range(num_test_envs):
                    test_env = test_env_names[idx_test]
                    test_return = test_stats[f"test_returns:{test_env}"]
                    if args.test_checkpoint:
                        if test_return > best_test_return[idx_test]:
                            checkpoint(checkpoint_best_test_model_path[idx_test])
                            logging.info(f"\nSaved best test checkpoint after update {j}: in {test_env}, {test_return} > {best_test_return[idx_test]}")
                            best_test_return[idx_test] = test_return
                    if args.test_solved_checkpoint and not solved_test_checkpoint_has_been_saved[idx_test]:
                        if test_return >= args.test_solved_return_threshold:
                            checkpoint(checkpoint_solved_test_model_path[idx_test])
                            logging.info(f"\nSaved solved test checkpoint after update {j}: in {test_env}, {test_return} >= {args.test_solved_return_threshold}")
                            solved_test_checkpoint_has_been_saved[idx_test] = True
                if args.early_stopping:
                    test_env_early_stopping = test_env_names[-1]
                    test_return_early_stopping = test_stats[f"test_returns:{test_env_early_stopping}"]
                    if test_return_early_stopping >= args.test_solved_return_threshold:
                        stop_training_due_to_test_solved = True
                if args.use_accel_paired:
                    adv_test_stats = evaluator.evaluate(train_runner.agents['adversary_agent'])
                    curr_keys = list(adv_test_stats.keys())
                    for curr_key in curr_keys:
                        adv_test_stats[f"advagent_{curr_key}"] = adv_test_stats[curr_key]
                        adv_test_stats.pop(curr_key, None)
                    stats.update(adv_test_stats)
            else:
                stats.update({k:None for k in evaluator.get_stats_keys()})

            update_end_time = timer()
            num_incremental_updates = 1 if j == 0 else args.log_interval
            sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            stats.update({'sps': sps})
            stats.update(test_stats) # Ensures sps column is always before test stats
            log_stats(stats)

        checkpoint_idx = getattr(train_runner, args.checkpoint_basis)

        if checkpoint_idx != last_checkpoint_idx:
            is_last_update = j == num_updates - 1 or (args.early_stopping and stop_training_due_to_test_solved)
            if is_last_update or \
                (train_runner.num_updates > 0 and checkpoint_idx % args.checkpoint_interval == 0):
                checkpoint(checkpoint_path, checkpoint_idx)
                logging.info(f"\nSaved checkpoint after update {j}")
                logging.info(f"\nLast update: {is_last_update}")
            elif train_runner.num_updates > 0 and args.archive_interval > 0 \
                and checkpoint_idx % args.archive_interval == 0:
                checkpoint(checkpoint_path, checkpoint_idx)
                logging.info(f"\nArchived checkpoint after update {j}")

        if save_screenshot:
            level_info = train_runner.sampled_level_info
            if args.env_name.startswith('BipedalWalker'):
                encodings = venv.get_level()
                df = bipedalwalker_df_from_encodings(args.env_name, encodings)
                if args.use_editor and level_info:
                    df.to_csv(os.path.join(
                        screenshot_dir, 
                        f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.csv"))
                else:
                    df.to_csv(os.path.join(
                        screenshot_dir, 
                        f'update{j}.csv'))
            else:
                venv.reset_agent()
                images = venv.get_images()
                if args.use_editor and level_info:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(
                            screenshot_dir, 
                            f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"), 
                        normalize=True, channels_first=False)
                else:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(screenshot_dir, f'update{j}.png'),
                        normalize=True, channels_first=False)
                plt.close()

        if args.early_stopping and stop_training_due_to_test_solved:
            break

    evaluator.close()
    venv.close()

    if display:
        display.stop()

    if args.verbose and args.early_stopping:
        if stop_training_due_to_test_solved:
            print(f"Training finished successfully due to early stopping (test return {test_return_early_stopping:.3f} >= {args.test_solved_return_threshold} in test env: {test_env_early_stopping}). Steps: {stats['steps']}")
        else:
            print(f"Training NOT finished successfully. Steps: {stats['steps']}")