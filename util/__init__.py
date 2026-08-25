# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import collections
import timeit
import random

import numpy as np
import torch
from torchvision import utils as vutils

from envs.registration import make as gym_make
from envs.registration import make_vec_linked as gym_make_vec_linked
from .make_agent import make_agent
from .filewriter import FileWriter
from envs.wrappers import ParallelAdversarialVecEnv, ParallelAdversarialLinkedVecEnv, VecMonitor, VecNormalize, \
    VecPreprocessImageWrapper, VecFrameStack, MultiGridFullyObsWrapper, CarRacingWrapper, TimeLimit


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def array_to_csv(a):
    return ','.join([str(v) for v in a])


def cprint(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path)
    path_tmp = f'{filename}_tmp{ext}'
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f'{filename}_{index}{ext}'
            shutil.copy(path, archive_path)


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_images(images, path=None, normalize=False, channels_first=False):
    if path is None:
        return

    if isinstance(images, (list, tuple)):
        images = torch.tensor(np.stack(images), dtype=torch.float)
    elif isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float)

    if normalize:
        images = images/255

    if not channels_first:
        if len(images.shape) == 4:
            images = images.permute(0,3,1,2)
        else:
            images = images.permute(2,0,1)

    grid = vutils.make_grid(images)
    vutils.save_image(grid, path)


def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]


def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)


def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ in ['Discrete', 'MultiDiscrete']
    else:
        return env.action_space.__class__.__name__ in ['Discrete', 'MultiDiscrete']


def _make_env(args):
    env_kwargs = {'seed': args.seed}
    if args.singleton_env:
        env_kwargs.update({
            'fixed_environment': True})
    if args.env_name.startswith('CarRacing'):
        env_kwargs.update({
            'n_control_points': args.num_control_points,
            'min_rad_ratio': args.min_rad_ratio,
            'max_rad_ratio': args.max_rad_ratio,
            'use_categorical': args.use_categorical_adv,
            'use_sketch': args.use_sketch,
            'clip_reward': args.clip_reward,
            'sparse_rewards': args.sparse_rewards,
            'num_goal_bins': args.num_goal_bins,
        })
    if args.env_name.startswith('Procgen'):
        env_kwargs.update({
            'resource_root': args.procgen_resource_root,
            'prebuilt_root': args.procgen_prebuilt_root,
        })

    if args.env_name.startswith('CarRacing'):
        # Hack: This TimeLimit sandwich allows truncated obs to be passed
        # up the hierarchy with all necessary preprocessing.
        env = gym_make(args.env_name, **env_kwargs)
        max_episode_steps = env._max_episode_steps
        reward_shaping = args.reward_shaping and not args.sparse_rewards
        assert max_episode_steps % args.num_action_repeat == 0
        return TimeLimit(CarRacingWrapper(env,
                grayscale=args.grayscale, 
                reward_shaping=reward_shaping,
                num_action_repeat=args.num_action_repeat,
                nstack=args.frame_stack,
                crop=args.crop_frame), 
            max_episode_steps=max_episode_steps//args.num_action_repeat)
    elif args.env_name.startswith('MultiGrid'):
        env = gym_make(args.env_name, **env_kwargs)
        if args.use_global_critic or args.use_global_policy:
            max_episode_steps = env._max_episode_steps
            env = TimeLimit(MultiGridFullyObsWrapper(env),
                max_episode_steps=max_episode_steps)
        return env
    elif args.env_name.startswith('MiniGrid'):
        env = gym_make(args.env_name, **env_kwargs)
        if args.use_global_critic or args.use_global_policy:
            raise NotImplementedError
        return env
    elif args.env_vectorization == 'env':
        env_kwargs.update({
            'num_procs': args.num_processes,
        })
        return gym_make_vec_linked(args.env_name, args.num_processes, **env_kwargs)
    else:
        return gym_make(args.env_name, **env_kwargs)


def create_parallel_env(args, adversary=True):
    is_multigrid = args.env_name.startswith('MultiGrid')
    is_car_racing = args.env_name.startswith('CarRacing')
    is_bipedalwalker = args.env_name.startswith('BipedalWalker')
    is_minigrid = args.env_name.startswith('MiniGrid')
    is_procgen = args.env_name.startswith('Procgen')

    make_fn = lambda: _make_env(args)

    venv = None
    if args.env_vectorization == 'dcd':
        venv = ParallelAdversarialVecEnv([make_fn]*args.num_processes, adversary=adversary)
    elif args.env_vectorization == 'env':
        venv = ParallelAdversarialLinkedVecEnv(make_fn, adversary=adversary)
    else:
        raise NotImplementedError

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns, gamma=args.gamma)

    obs_key = None
    scale = None
    transpose_order = [2,0,1] # Channels first
    if is_multigrid or is_minigrid:
        obs_key = 'image'
        scale = 10.0

    if is_car_racing:
        ued_venv = VecPreprocessImageWrapper(venv=venv) # move to tensor

    if is_bipedalwalker:
        transpose_order = None

    if is_procgen:
        obs_key = 'image'
        scale = 255.0

    venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
            transpose_order=transpose_order, scale=scale)

    if is_multigrid or is_bipedalwalker or is_minigrid or is_procgen:
        ued_venv = venv

    if args.singleton_env:
        seeds = [args.seed]*args.num_processes
    else:
        if args.offset_seed:
            # This can support up to 1000 random seeds (0-999) and up to 10000 num_processes (0-9999) before there are seed overlaps
            seeds = [int(args.seed)*10000 + i for i in range(args.num_processes)]
        else:
            seeds = [i for i in range(args.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv


def is_dense_reward_env(env_name):
    if env_name.startswith('CarRacing'):
        return True
    else:
        return False


def make_plr_args(args, obs_space, action_space):
    return dict( 
        seeds=[], 
        obs_space=obs_space, 
        action_space=action_space, 
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        replay_prob=args.level_replay_prob, 
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
        sample_full_distribution=args.train_full_distribution,
        seed_buffer_size=args.level_replay_seed_buffer_size,
        seed_buffer_priority=args.level_replay_seed_buffer_priority,
        use_dense_rewards=is_dense_reward_env(args.env_name),
        gamma=args.gamma
    )

def make_param_distrib_args(args, device=None):
    param_distrib_args = dict(
        strategy=args.param_distrib_strategy,
        env_name=args.env_name,
        num_processes=args.num_processes,
        seed=args.seed,
    )
    if args.param_distrib_config is not None:
        param_distrib_args.update(args.param_distrib_config)

    eval_config = dict(
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
    )
    param_distrib_args['eval_config'] = eval_config
    param_distrib_args['deterministic_eval'] = args.deterministic_test_evaluation

    eval_env_kwargs = {}
    if 'Procgen' in args.test_env_names:
        if args.env_vectorization == 'env':
            eval_env_kwargs['num_procs'] = args.test_num_processes
        eval_env_kwargs['resource_root'] = args.procgen_resource_root
        eval_env_kwargs['prebuilt_root'] = args.procgen_prebuilt_root
    param_distrib_args['eval_env_kwargs'] = eval_env_kwargs

    return param_distrib_args
