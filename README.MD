# CURATE

This is the official implementation of [CURATE (TMLR 2026)](https://montrealrobotics.ca/curate/), an automatic curriculum learning algorithm for reinforcement learning agents.
This implementation also features an example integration with the [Procgen Curriculum Suite](https://github.com/montrealrobotics/procgen-curriculum), which was also introduced in the CURATE work.

# Installation

The following instructions have been verified for Linux Ubuntu 24.04.2.

## Downloading the codebase
```
cd path/to/ws/
git clone --recursive git@github.com:montrealrobotics/curate-code.git
cd curate-code/
```

## uv
[uv](https://docs.astral.sh/uv/) is a package and project manager. First, install and setup uv using the instructions:
[https://docs.astral.sh/uv/getting-started/](https://docs.astral.sh/uv/getting-started/)


(Optional) If you would like to pin a particular version of Python, please enter it here. By default, Python 3.8.19 is used, which is the version we used for testing.
```
uv python pin 3.8.19
```

To install using uv, please run:
```
uv sync
```

# Example commands

The following commands are lightly edited from the commands we used to run CURATE experiments on a SLURM HPC cluster (e.g., fewer environment steps). The hyperparameters used here are the same as in the paper.

To try running CURATE for MiniGrid MultiRoom:
```
export EXPERIMENT_SEED=0; uv run python -m train --xpid=MiniGrid-MultiRoom-N1to4-S4to7-G13-T80-Adversarial-v0-testN4to4S4to7G13T80-nostop-Curate-tsrt0p3t0p7-keeptarget --env_name=MiniGrid-MultiRoom-N1to4-S4to7-G13-T80-Adversarial-v0 --use_gae=True --gamma=0.995 --gae_lambda=0.95 --seed=-99 --seed_variable=EXPERIMENT_SEED --offset_seed=True --append_seed_to_xpid=True --seed_variable_offset=0 --recurrent_arch=lstm --recurrent_agent=True --recurrent_adversary_env=False --recurrent_hidden_size=256 --lr=0.0001 --num_steps=256 --num_processes=32 --env_vectorization=dcd --num_env_steps=25000000 --ppo_epoch=5 --num_mini_batch=1 --entropy_coef=0.0 --value_loss_coef=0.5 --clip_param=0.2 --clip_value_loss=True --adv_entropy_coef=0.0 --max_grad_norm=0.5 --normalize_returns=False --algo=ppo --ued_algo=parameter_distribution --param_distrib_strategy=Curate --param_distrib_config='{"min_params": 1, "max_params": 4, "num_param_steps": 4, "return_threshold": 0.3, "params_mean_init": "auto", "params_var_init": "auto", "min_params_var_diag": 0.1, "params_mean_shift_on_solve": "auto", "params_var_on_update": "auto", "eval_env_template": "MiniGrid-MultiRoom-N%P1to%P1-S4to7-G13-T80-v0", "reps_rounds": 2, "reps_rounds_init": 4, "samples_per_reps_round": 8, "reps_min_temperature": 0.05, "coef_param_reg_init": "auto", "lp_penalty": "auto", "use_lp_penalty_during_init": false, "lp_penalty_start_params_mode": "learned", "num_mc_samples": 5000000, "min_update_params_ctr": 16, "force_update_params_ctr": 128, "num_processes_eval": 4, "num_episodes_eval": 16, "keep_solved_target_task_returns": true}' --use_plr=True --level_replay_prob=0.0 --level_replay_rho=0.5 --level_replay_seed_buffer_size=4000 --level_replay_score_transform=rank --level_replay_temperature=0.3 --staleness_coef=0.3 --no_exploratory_grad_updates=False --use_editor=False --level_editor_prob=0 --level_editor_method=random --num_edits=0 --base_levels=batch --log_interval=1 --screenshot_interval=1000 --log_grad_norm=False --handle_timelimits=True --checkpoint_basis=student_grad_updates --level_replay_strategy=positive_value_loss --test_env_names=MiniGrid-MultiRoom-N4to4-S4to7-G13-T80-v0 --test_num_episodes=128 --test_num_processes=32 --early_stopping=False --test_interval=1 --deterministic_test_evaluation=False --test_seed=-99 --test_stagger_seeds=True --test_offset_seed=True --test_checkpoint=True --test_solved_checkpoint=True --test_solved_return_threshold=0.7 --log_dir=/path/to/curate/experiments/multiroom-n1to4s4to7/curate-tsrt0p3t0p7-keeptarget/seeds/multiroom-n1to4s4to7-nostop-curate-tsrt0p3t0p7-keeptarget --log_action_complexity=True --log_plr_buffer_stats=True --log_replay_complexity=True --reject_unsolvable_seeds=False --verbose --assert_cuda=True --assert_no_prior_xpid=True
```

To try running CURATE for CoinRun (`procgen_resource_root` and `procgen_prebuilt_root` can also be used to pass a prebuilt version of Procgen):
```
export EXPERIMENT_SEED=0; uv run python -m train --xpid=Procgen-CoinRun-Easy-D1to3-S1to5-Adversarial-v0-testD3to3S5to5-nostop-Curate-tsrt8-keeptarget --env_name=Procgen-CoinRun-Easy-D1to3-S1to5-Adversarial-v0 --use_gae=True --gamma=0.999 --gae_lambda=0.95 --seed=-99 --seed_variable=EXPERIMENT_SEED --offset_seed=True --append_seed_to_xpid=True --seed_variable_offset=0 --recurrent_arch=gru --recurrent_agent=False --recurrent_adversary_env=False --recurrent_hidden_size=1 --lr=0.0005 --num_steps=1024 --num_processes=64 --env_vectorization=env --num_env_steps=50000000 --ppo_epoch=3 --num_mini_batch=32 --entropy_coef=0.01 --value_loss_coef=0.5 --clip_param=0.2 --clip_value_loss=True --adv_entropy_coef=0.0 --max_grad_norm=0.5 --normalize_returns=True --algo=ppo --ued_algo=parameter_distribution --param_distrib_strategy=Curate --param_distrib_config='{"min_params": [1, 1], "max_params": [3, 5], "num_param_steps": [3, 5], "return_threshold": 8.0, "params_mean_init": "auto", "params_var_init": "auto", "min_params_var_diag": [0.1, 0.1], "params_mean_shift_on_solve": "auto", "params_var_on_update": "auto", "eval_env_template": "Procgen-CoinRun-Easy-D%P1to%P1-S%P2to%P2-v0", "reps_rounds": 2, "reps_rounds_init": 4, "samples_per_reps_round": 16, "reps_min_temperature": 0.05, "coef_param_reg_init": "auto", "lp_penalty": "auto", "use_lp_penalty_during_init": false, "lp_penalty_start_params_mode": "learned", "num_mc_samples": 5000000, "min_update_params_ctr": 16, "force_update_params_ctr": 128, "num_processes_eval": 16, "num_episodes_eval": 32, "parallel_eval_workers": 4, "keep_solved_target_task_returns": true, "params_mean_shift_mode": "directed"}' --use_plr=True --level_replay_prob=0.0 --level_replay_rho=0.5 --level_replay_seed_buffer_size=4000 --level_replay_score_transform=rank --level_replay_temperature=0.3 --staleness_coef=0.3 --no_exploratory_grad_updates=False --use_editor=False --level_editor_prob=0 --level_editor_method=random --num_edits=0 --base_levels=batch --log_interval=1 --screenshot_interval=0 --log_grad_norm=False --handle_timelimits=True --checkpoint_basis=student_grad_updates --level_replay_strategy=positive_value_loss --test_env_names=Procgen-CoinRun-Easy-D3to3-S5to5-v0 --test_num_episodes=128 --test_num_processes=64 --early_stopping=False --test_interval=1 --deterministic_test_evaluation=False --test_seed=-99 --test_stagger_seeds=True --test_offset_seed=True --test_checkpoint=True --test_solved_checkpoint=True --test_solved_return_threshold=8.0 --log_dir=/path/to/curate/experiments/coinrun-easy-d1to3s1to5/curate-tsrt8-keeptarget-pmsosdir/seeds/coinrun-easy-d1to3s1to5-nostop-curate-tsrt8-keeptarget-pmsosdir --log_action_complexity=True --log_plr_buffer_stats=True --log_replay_complexity=True --reject_unsolvable_seeds=False --verbose --assert_cuda=True --assert_no_prior_xpid=True
```

To try running CURATE for BipedalWalker:
```
export EXPERIMENT_SEED=0; uv run python -m train --xpid=BipedalWalker-Adversarial-v0-test7envs-nostop-Curate-tsrt200-ocs0p25solved --env_name=BipedalWalker-Adversarial-v0 --use_gae=True --gamma=0.99 --gae_lambda=0.9 --seed=-99 --seed_variable=EXPERIMENT_SEED --offset_seed=True --append_seed_to_xpid=True --seed_variable_offset=0 --recurrent_arch=lstm --recurrent_agent=False --recurrent_adversary_env=False --recurrent_hidden_size=1 --use_global_critic=False --lr=0.0003 --num_steps=2048 --num_processes=16 --num_env_steps=1000000000 --ppo_epoch=5 --num_mini_batch=32 --entropy_coef=0.001 --value_loss_coef=0.5 --clip_param=0.2 --clip_value_loss=False --adv_entropy_coef=0.01 --max_grad_norm=0.5 --normalize_returns=True --use_lstm=False --algo=ppo --ued_algo=parameter_distribution --param_distrib_strategy=Curate --param_distrib_config='{"min_params": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "max_params": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 9.0], "num_param_steps": [101, 101, 101, 51, 51, 51, 51, 91], "return_threshold": 200.0, "params_mean_init": "auto", "params_var_init": "auto", "min_params_var_diag": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "params_mean_shift_on_solve": "auto", "params_mean_shift_on_solve_auto_divisor": 20.0, "params_var_on_update": "auto", "params_var_on_update_auto_divisor": 10.0, "eval_env_template": "BipedalWalker-Params-v0", "reps_rounds": 2, "reps_rounds_init": 8, "samples_per_reps_round": 64, "reps_min_temperature": 0.05, "coef_param_reg_max_learning_penalty": 10.0, "coef_param_reg_init": "auto", "lp_penalty_max_learning_penalty": 10.0, "lp_penalty": "auto", "use_lp_penalty_during_init": false, "lp_penalty_start_params_mode": "learned", "min_update_params_ctr": 128, "force_update_params_ctr": 256, "parallel_eval_workers": 8, "keep_solved_target_task_returns": true, "params_mean_shift_mode": "directed", "off_curriculum_strategy": "solved", "off_curriculum_strategy_prob": 0.25, "off_curriculum_sampling": "deterministic", "off_curriculum_return_set": "on_curriculum"}' --use_plr=True --level_replay_prob=0.0 --level_replay_rho=0.5 --level_replay_seed_buffer_size=1000 --level_replay_score_transform=rank --level_replay_temperature=0.1 --staleness_coef=0.5 --no_exploratory_grad_updates=False --use_editor=False --level_editor_prob=0 --level_editor_method=random --num_edits=0 --base_levels=batch --use_accel_paired=False --accel_paired_score_function=paired --use_behavioural_cloning=False --kl_loss_coef=0.0 --kl_update_step=1 --use_kl_only_agent=False --log_interval=1 --screenshot_interval=200 --log_grad_norm=True --checkpoint_basis=student_grad_updates --archive_interval=5000 --handle_timelimits=True --level_replay_strategy=positive_value_loss --test_env_names=BipedalWalker-v3,BipedalWalkerHardcore-v3,BipedalWalker-Med-Stairs-v0,BipedalWalker-Med-PitGap-v0,BipedalWalker-Med-StumpHeight-v0,BipedalWalker-Med-Roughness-v0,BipedalWalker-Max-v0 --log_dir=/path/to/curate/experiments/bipedalwalker-full/curate-tsrt200-ocs0p25solved-keeptarget-pmsosdir-cpu/seeds/bipedalwalker-full-nostop-curate-tsrt200-ocs0p25solved-keeptarget-pmsosdir --test_interval=100 --test_num_episodes=10 --test_num_processes=2 --early_stopping=False --deterministic_test_evaluation=False --test_checkpoint=True --test_solved_checkpoint=True --test_solved_return_threshold=300.0 --test_aggregation=True --log_plr_buffer_stats=True --log_replay_complexity=True --log_action_complexity=False --verbose --no_cuda=True --assert_no_prior_xpid=True
```

# Citation

If you find CURATE useful or related to your research, please consider citing our work. Thank you. 

```
@article{lee2026curate,
  title={CURATE: Automatic Curriculum Learning for Reinforcement Learning Agents through Competence-Based Curriculum Policy Search in Structured Task Spaces},
  author={Lee, Tabitha Edith and Ke, Nan Rosemary and Patil, Sarvesh and Dahmani, Annya and Yiu, Eunice and Saleh, Esra'a and Gopnik, Alison and Kroemer, Oliver and Berseth, Glen},
  journal={Transactions on Machine Learning Research (TMLR)},
  month={August},
  year={2026},
  organization={TMLR},
  url={https://openreview.net/forum?id=DlnvWfoIgv},
}
```

# Acknowledgments

We thank [M. Jiang and M. Dennis et al. (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html) for open-sourcing the DCD repository, which we extended in the CURATE work.
For more information about this repository, please refer to the [original DCD repository](https://github.com/facebookresearch/dcd).
