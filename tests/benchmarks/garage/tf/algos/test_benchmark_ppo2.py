"""A regression test over PPO Algorithms."""

import datetime
import multiprocessing
import os.path as osp
import random

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.logger import configure
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf
import torch

from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline as TF_GMB
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import PPO as PyTorch_PPO
from garage.torch.baselines import GaussianMLPBaseline as PyTorch_GMB
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper
from tests import helpers as Rh
from tests.fixtures import snapshot_config
from tests.wrappers import AutoStopEnv

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'center_adv': True,
    'learning_rate': 3e-4,
    'lr_clip_range': 0.1,
    'gae_lambda': 0.95,
    'discount': 0.99,
    'n_epochs': 400,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_trials': 4,
    'optimize_minibatch_size': 128,  # 32 for tf/ppo
    'optimize_epochs': 10
}


# pylint: disable=too-few-public-methods
class TestBenchmarkPPO:
    """A regression test over PPO Algorithms.
    (garage-PyTorch-PPO, garage-TensorFlow-PPO, and baselines-PPO2)

    It get Mujoco1M benchmarks from baselines benchmark, and test each task in
    its trial times on garage model and baselines model. For each task,
    there will
    be `trial` times with different random seeds. For each trial, there will
    be two
    log directories corresponding to baselines and garage. And there will be
    a plot
    plotting the average return curve from baselines and garage.
    """

    @staticmethod
    @pytest.mark.huge
    def test_benchmark_ppo():
        """Compare benchmarks between garage and baselines.

        Returns:

        """
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/ppo/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
        # Hopper-v2, Walker2d-v2, Reacher-v2, Swimmer-v2
        # for task in [env for env in mujoco1m['tasks'] if env['env_id'] in ['HalfCheetah-v2']]:
            env_id = task['env_id']

            env = gym.make(env_id)

            seeds = random.sample(range(100), hyper_parameters['n_trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))

            garage_pytorch_LFB_csvs = []
            garage_pytorch_GMB_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_pytorch_dir_LFB = trial_dir + '/garage/pytorch/LFB'
                garage_pytorch_dir_GMB = trial_dir + '/garage/pytorch/GMB'

                env.reset()
                garage_pytorch_csv_GMB = run_garage_pytorch_GMB(
                    env, seed, garage_pytorch_dir_GMB)

                env.reset()
                garage_pytorch_csv_LFB = run_garage_pytorch_LFB(
                    env, seed, garage_pytorch_dir_LFB)

                garage_pytorch_LFB_csvs.append(garage_pytorch_csv_LFB)
                garage_pytorch_GMB_csvs.append(garage_pytorch_csv_GMB)

            env.close()

            benchmark_helper.plot_average_over_trials(
                [garage_pytorch_LFB_csvs, garage_pytorch_GMB_csvs],
                ['Evaluation/AverageReturn', 'Evaluation/AverageReturn'],
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='Evaluation/AverageReturn',
                names=['garage-PyTorch-LFB', 'garage-PyTorch-GMB'],
            )

            result_json[env_id] = benchmark_helper.create_json(
                [garage_pytorch_LFB_csvs, garage_pytorch_GMB_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=[
                    'Evaluation/Iteration',
                    'Evaluation/Iteration',
                ],
                ys=['Evaluation/AverageReturn', 'Evaluation/AverageReturn'],
                factors=[hyper_parameters['batch_size']] * 2,
                names=['garage-PT-LFB', 'garage-PT-GMB'])

        Rh.write_file(result_json, 'PPO')


def run_garage_pytorch_GMB(env, seed, log_dir):
    """Create garage PyTorch PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    env = TfEnv(normalize(env))

    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    baseline = PyTorch_GMB(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        optimizer_args=dict(lr=hyper_parameters['learning_rate']))

    algo = PyTorch_PPO(
        env_spec=env.spec,
        policy=policy,
        baseline=baseline,
        optimizer=torch.optim.Adam,
        policy_lr=hyper_parameters['learning_rate'],
        max_path_length=hyper_parameters['max_path_length'],
        discount=hyper_parameters['discount'],
        gae_lambda=hyper_parameters['gae_lambda'],
        center_adv=hyper_parameters['center_adv'],
        lr_clip_range=hyper_parameters['lr_clip_range'],
        minibatch_size=hyper_parameters['optimize_minibatch_size'],
        max_optimization_epochs=hyper_parameters['optimize_epochs'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])

    dowel_logger.remove_all()

    return tabular_log_file


def run_garage_pytorch_LFB(env, seed, log_dir):
    """Create garage PyTorch PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    env = TfEnv(normalize(env))

    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PyTorch_PPO(
        env_spec=env.spec,
        policy=policy,
        baseline=baseline,
        optimizer=torch.optim.Adam,
        policy_lr=hyper_parameters['learning_rate'],
        max_path_length=hyper_parameters['max_path_length'],
        discount=hyper_parameters['discount'],
        gae_lambda=hyper_parameters['gae_lambda'],
        center_adv=hyper_parameters['center_adv'],
        lr_clip_range=hyper_parameters['lr_clip_range'],
        minibatch_size=hyper_parameters['optimize_minibatch_size'],
        max_optimization_epochs=hyper_parameters['optimize_epochs'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])

    dowel_logger.remove_all()

    return tabular_log_file
