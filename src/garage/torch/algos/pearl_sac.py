# exisitng PEARL uses old version of rlkit SAC (2 q-functions and value function)
# new SAC uses 2 q-functions, and value function is replaced by getting action from
# policy then calculating q value like in DDPG

from collections import OrderedDict
import copy
import time

from dowel import logger, tabular
import gtimer as gt
import numpy as np
import torch

from garage.experiment.snapshotter import Snapshotter
from garage.misc import eval_util
from garage.replay_buffer.multi_task_replay_buffer import MultiTaskReplayBuffer
from garage.sampler.in_place_sampler import InPlaceSampler
import garage.torch.utils as tu


class PEARLSAC:
    def __init__(self,
                 env,
                 train_tasks,
                 eval_tasks,
                 latent_dim,
                 nets,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 vf_lr=1e-3,
                 context_lr=1e-3,
                 kl_lambda=1.,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 optimizer_class=torch.optim.Adam,
                 recurrent=False,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,
                 sparse_rewards=False,

                 soft_target_tau=1e-2,

                 meta_batch=64,
                 num_iterations=100,
                 num_train_steps_per_itr=1000,
                 num_initial_steps=100,
                 num_tasks_sample=100,
                 num_steps_prior=100,
                 num_steps_posterior=100,
                 num_extra_rl_steps_posterior=100,
                 num_evals=10,
                 num_steps_per_eval=1000,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=1000,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 reward_scale=1,
                 num_exp_traj_eval=1,
                 update_post_train=1,
                 eval_deterministic=True,
                 render=False,
                 save_replay_buffer=False,
                 save_algorithm=False,
                 save_environment=False,
                 render_eval_paths=False,
                 dump_eval_paths=False,
                 plotter=None,

                 ):

        """
        # off policy params
        #self.env = env
        #self.policy = policy
        #self.qf = qf
        self.replay_buffer = replay_buffer
        self.n_epoch_cycles = n_epoch_cycles
        self.n_train_steps = n_train_steps
        self.buffer_batch_size = buffer_batch_size
        self.use_target = use_target
        self.discount = discount
        self.min_buffer_size = min_buffer_size
        self.rollout_batch_size = rollout_batch_size
        self.reward_scale = reward_scale
        self.evaluate = False
        #self.input_include_goal = input_include_goal
        self.smooth_return = smooth_return
        self.max_path_length = max_path_length
        self.es = exploration_strategy
        self.sampler_cls = OffPolicyVectorizedSampler
        """

        # meta params
        self.env = env
        self.policy = nets[0]
        self.exploration_policy = nets[0] # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        # same as buffer_batch_size?
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        # same as min_buffer_size?
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.sampler = InPlaceSampler(
            env=env,
            policy=nets[0],
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        #self._current_path_builder = PathBuilder()
        self._exploration_paths = []

        # pearl params
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()
        self.vib_criterion = torch.nn.MSELoss()
        self.l2_reg_criterion = torch.nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = copy.deepcopy(self.vf)

        self.policy_optimizer = optimizer_class(
            self.policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.networks[0].parameters(),
            lr=context_lr,
        )

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self, runner):
        '''
        meta-training loop
        '''
        #params = self.get_epoch_snapshot(-1)
        #self._snapshotter.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        #self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for _ in runner.step_epochs():
            #self.training_mode(True)
            for itr in range(self.num_iterations):
                if itr == 0:
                    print('collecting initial pool of data for train and eval')
                    # temp for evaluating
                    for idx in self.train_tasks:
                        self.task_idx = idx
                        self.env.reset_task(idx)
                        self.collect_data(self.num_initial_steps, 1, np.inf)
                
                # Sample data from train tasks.
                for i in range(self.num_tasks_sample):
                    idx = np.random.randint(len(self.train_tasks))
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.enc_replay_buffer.task_buffers[idx].clear()

                    # collect some trajectories with z ~ prior
                    if self.num_steps_prior > 0:
                        self.collect_data(self.num_steps_prior, 1, np.inf)
                    # collect some trajectories with z ~ posterior
                    if self.num_steps_posterior > 0:
                        self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                    if self.num_extra_rl_steps_posterior > 0:
                        self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)

                # Sample train tasks and compute gradient updates on parameters.
                for train_step in range(self.num_train_steps_per_itr):
                    indices = np.random.choice(self.train_tasks, self.meta_batch)
                    self._do_training(indices)
                    self._n_train_steps_total += 1

                gt.stamp('train')

                #self.training_mode(False)

                # eval
                self._try_to_eval(itr)
                gt.stamp('eval')

                self._end_epoch()

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        self.policy.reset_belief()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.policy.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        #logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            #params = self.get_epoch_snapshot(epoch)
            #self._snapshotter.save_itr_params(epoch, params)
            '''
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys
            '''
            tabular.record(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            tabular.record(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            tabular.record(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            tabular.record('Train Time (s)', train_time)
            tabular.record('(Previous) Eval Time (s)', eval_time)
            tabular.record('Sample Time (s)', sample_time)
            tabular.record('Epoch Time (s)', epoch_time)
            tabular.record('Total Train Time (s)', total_time)

            tabular.record("Epoch", epoch)
            #logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, policy, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        policy.set_num_steps_total(self._n_env_steps_total)
        return policy.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()
    
    ##### Snapshotting utils #####
    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.policy.reset_belief()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.policy.infer_posterior(self.policy.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        #if self.dump_eval_paths:
        #    logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.policy.reset_belief()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            #logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.policy.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.policy.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        #logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        #logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            tabular.record(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.policy.networks + [self.policy] + [self.qf1, self.qf2, self.vf, self.target_vf]

    """
    def training_mode(self, mode):
        print("***** ", self.networks)
        for net in self.networks:
            net.train(mode)
    """

    def to(self, device=None):
        if device == None:
            device = tu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [tu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [tu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)
        # zero out context and hidden encoder state
        self.policy.reset_belief(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)
            # stop backprop
            self.policy.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        for target_param, param in zip(self.target_vf.parameters(), self.vf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) \
                    + param.data * self.soft_target_tau
            )

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates

        #with torch.no_grad():
        target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            print("lw_debug: take_step after kl_div")
            kl_loss = self.kl_lambda * kl_div
            print("lw_debug: take_step after kl_loss")
            #kl_loss.backward(retain_graph=True)
            kl_loss.backward()
            print("lw_debug: take_step after kl_loss backward")

        print("lw_debug: take_step after context_optimizer")
        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        print("lw_debug: take_step before qf_loss.backward")
        print(qf_loss)

        qf_loss.backward()
        print("lw_debug: take_step after qf_loss.backward")
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)


        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        print("lw_debug: take_step after policy loss optimize")
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(tu.get_numpy(self.policy.z_means[0])))
                z_sig = np.mean(tu.get_numpy(self.policy.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = tu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = tu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(tu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(tu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(tu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Q Predictions',
                tu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'V Predictions',
                tu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Log Pis',
                tu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Policy mu',
                tu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'Policy log std',
                tu.get_numpy(policy_log_std),
            ))
    """
    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.policy.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.policy.context_encoder.state_dict(),
        )
        return snapshot
    """

