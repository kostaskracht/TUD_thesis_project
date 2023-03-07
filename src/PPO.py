import time

import gym
import thesis_env
import os
import math
import shutil
import numpy as np
import yaml
import torch as th
from torch import nn
from scipy.signal import lfilter
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from copy import copy


class MindmapRolloutBuffer:
    """
    Buffer that stores the trajectory of the current episode. After the end of the episode it
    calculates the advantage.

    :param num_states (int): Number of discrete states per component
    :param num_actions (int): Number of environmental actions
    :param num_components (int): Number of discrete components
    :param timesteps (int): Number of timesteps to in the environment
    :param gamma (float): Discount factor
    :param lam (float): Lambda parameter

    """

    def __init__(self, num_states, num_actions, num_components, timesteps, num_objectives, gamma=0.95, lam=0.95):
        # Buffer initialization
        self.num_components = num_components
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # The observation dimension equals to components*(states_num)
        # IRI is stationary, so state embedding only consists of IRI
        self.observation_dimensions = self.num_components * (self.num_states) + 1
        self.size = timesteps
        self.gamma = gamma
        self.lam = lam

        # Reset buffer
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.observation_buffer = np.zeros((self.size, self.observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros((self.size, self.num_components), dtype=np.float32)
        self.advantage_buffer = np.zeros(self.size, dtype=np.float32)
        self.reward_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
        self.return_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
        self.value_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
        self.logprobability_buffer = np.zeros(self.size, dtype=np.float32)
        self.counter, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        """
        Append one step of agent-environment interaction
        :param observation: ndarray - Believe matrix (num_components x (num_states + 1))
        :param action: ndarray - Actions for the specific timestep (num_components)
        :param reward: float - Reward for the specific timestep
        :param value: float - Value for the specific timestep
        :param logprobability: ndarray - Log probabilities for ___TODO (num_components x num_states)
        """
        self.observation_buffer[self.counter] = observation
        self.action_buffer[self.counter] = action
        self.reward_buffer[self.counter] = reward
        self.value_buffer[self.counter] = value
        self.logprobability_buffer[self.counter] = logprobability.detach()
        self.counter += 1

    def finish_trajectory(self, last_value=0, w_rewards=None):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.counter)
        rewards = np.append(self.reward_buffer[path_slice], [[last_value] * self.num_objectives], axis=0)
        # rewards_dot = np.einsum('ij,j->i', rewards, w_rewards)

        values = np.append(self.value_buffer[path_slice], [[last_value] * self.num_objectives], axis=0)
        values_dot = np.einsum('ij,j->i', values, w_rewards)
        # deltas = (rewards_dot + self.gamma * np.roll(values_dot, -1) - values_dot)[:-1]

        # self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(rewards, self.gamma)[:-1]
        returns_dot = np.einsum('ij,j->i', self.return_buffer[path_slice], w_rewards)

        # Simplification
        self.advantage_buffer[path_slice] = returns_dot - values_dot[:-1]
        self.trajectory_start_index = self.counter

    def get(self):
        # Get all data of the buffer
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        # This is equivalent to:
        # summ = [x[-1]]
        # for val in reversed(x[:-1]):
        #     summ.append(summ[-1] * discount + val)
        # return summ[::-1]


class MindmapActor(nn.Module):
    def __init__(self, num_components, num_states, num_actions, num_objectives, device, timestamp, optimizer, lr,
                 lr_min, lr_decay_step, lr_decay_episode_perc, n_epochs, architecture,
                 checkpoint_dir,
                 checkpoint_suffix="actor"):
        super(MindmapActor, self).__init__()

        # Configure save directory
        self.checkpoint_folder = f"{checkpoint_dir}{timestamp}/"
        self.checkpoint_suffix = checkpoint_suffix
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        # Initialize environment values
        self.num_components = num_components
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.n_epochs = n_epochs
        self.device = device

        # Setup network input and output
        # Since IRI is stationary, we don't need to add the time in the state embedding
        self.input_dim = (self.num_states) * self.num_components + 1
        # self.input_dim = (self.num_states + 1) * self.num_components + 1
        if self.checkpoint_suffix == "actor":
            self.output_dim = self.num_components * self.num_actions
        else:
            self.output_dim = self.num_objectives

        # Initialize the optimizer parameters
        self.optimizer_name = optimizer
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay_step = lr_decay_step
        self.lr_decay_episode_perc = lr_decay_episode_perc
        self.decay_epochs = self.n_epochs * self.lr_decay_episode_perc
        root = math.ceil(self.decay_epochs / self.lr_decay_step)
        self.gamma_decay = (self.lr_min / self.lr) ** (1 / root)

        self.supported_activation_fns = {"relu": nn.ReLU()}
        self.arch = architecture

        # get network layer dimension pairs (input-output for each layer)
        self.net_dims = self.get_net_dims()

        # Add network layers
        self.layers = nn.ModuleList()
        dim_count = 0
        for layer in self.arch:
            if isinstance(layer, int):
                # Adding a linear layer
                self.layers.append(nn.Linear(self.net_dims[dim_count][0], self.net_dims[
                    dim_count][1]))
                dim_count += 1
            else:
                # Adding an activation function
                try:
                    self.layers.append(copy(self.supported_activation_fns[layer]))
                except ValueError(f"Activation function {layer} currently not supported!") as e:
                    print(e)

        # Add output layer
        self.layers.append(nn.Linear(self.net_dims[-1][0], self.net_dims[-1][1]))

        # Initialize optimizer
        opt1 = f"optim.{self.optimizer_name}"
        opt2 = f"(self.parameters(), lr=self.lr)"
        self.optimizer = eval(opt1 + opt2)
        self.to(self.device)

        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.gamma_decay)

    def get_net_dims(self):
        net_dims = [self.input_dim] + \
                   [num for num in self.arch if isinstance(num, int)] + \
                   [self.output_dim]

        return list(zip(net_dims, net_dims[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def transform_with_softmax(self, actions):
        dist = th.reshape(actions, shape=(-1, self.num_components, self.num_actions))
        soft = nn.Softmax(dim=2)
        dist = soft(dist)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self, episode):
        checkpoint_path = f"{self.checkpoint_folder}ep{episode}_{self.checkpoint_suffix}.pth"
        th.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir, checkpoint_ep, reuse_mode):
        checkpoint_path = f"{checkpoint_dir}ep{checkpoint_ep}_{self.checkpoint_suffix}.pth"
        dict_to_load = th.load(checkpoint_path, map_location=self.device)
        if reuse_mode == "partial":
            # Don't load the final layer!
            for weight in list(self.state_dict().keys())[-2:]:
                dict_to_load[weight] = self.state_dict()[weight]

        self.load_state_dict(dict_to_load)


class MindmapCritic(MindmapActor):
    def __init__(self, num_components, num_states, num_actions, num_objectives, device, timestamp, optimizer, lr,
                 lr_min, lr_decay_step, lr_decay_episode_perc, n_epochs, architecture,
                 checkpoint_dir,
                 checkpoint_suffix="critic"):
        super(MindmapCritic, self).__init__(num_components, num_states, num_actions, num_objectives,
                                            device, timestamp, optimizer, lr, lr_min,
                                            lr_decay_step, lr_decay_episode_perc, n_epochs,
                                            architecture,
                                            checkpoint_dir, checkpoint_suffix)


class MindmapPPO:
    """
    The main class containing the MindmapPPO algorithm
    """

    def __init__(self, param_file="src/model_params.yaml", quiet=False):

        # Get timestamp of the execution
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(np.random.choice(1000)).zfill(3)
        self.output_dir = f"outputs/model_outputs/{self.timestamp}/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load parameters
        self.quiet = quiet
        self.param_file = param_file
        self.param_dict = self._load_yaml_file()

        # Create class attributes from parameters dictionary
        self._load_params()

        self.env = gym.make(self.env_name, quiet=True)
        self.env.reset()

        if self.seed:
            th.manual_seed(self.seed)
            np.random.seed(self.seed)

        # output the yaml files to the output dict
        shutil.copy(self.env.param_file, self.output_dir)
        shutil.copy(self.param_file, self.output_dir)

        # Initialize data structures to track reward and actions throughout execution
        self.total_rewards = []
        self.total_actions = []
        self.total_rewards_test = []
        self.total_actions_test = []

        # Initialize Rollout buffer
        self.buffer = MindmapRolloutBuffer(self.env.num_states_iri, self.env.num_actions,
                                           self.env.num_components, self.env.timesteps, self.env.num_objectives,
                                           self.gamma, self.lam)

        # Initialize actor and critic networks
        self.actor = MindmapActor(self.env.num_components, self.env.num_states_iri,
                                  self.env.num_actions, self.env.num_objectives, self.device, self.timestamp,
                                  self.optimizer_act, self.lr_act, self.lr_act_min,
                                  self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                  self.actor_arch, self.checkpoint_dir)
        self.critic = MindmapCritic(self.env.num_components, self.env.num_states_iri,
                                    self.env.num_actions, self.env.num_objectives, self.device, self.timestamp,
                                    self.optimizer_crit, self.lr_crit, self.lr_crit_min,
                                    self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                    self.critic_arch, self.checkpoint_dir, checkpoint_suffix="crit")

        # Initialize tensorboard logger
        self.writer = SummaryWriter(f"runs/{self.timestamp}")
        self.log = {"train_returns": "Returns/Train returns over epochs",
                    "test_returns": "Returns/Test returns over epochs",
                    "lr_actor": "Actor/Actor Learning Rate over epochs",
                    "lr_critic": "Critic/Critic Learning Rate over epochs",
                    "actor_loss": "Actor/Actor Loss over epochs",
                    "critic_loss": "Critic/Critic Loss over epochs",
                    "critic_value": "Critic/Value"}

        self.log_at_start()

        print(f"Current execution timestamp is {self.timestamp}")

    def sample_action(self, observation, ep):
        observation_tensor = th.tensor(np.array(observation), dtype=th.float).to(self.device)

        epsilon = 10
        if self.use_exploration_rate:
            epsilon0 = .0
            epsilon1 = .0
            e_perc = 0.3
            epsilon = np.max([(- epsilon1 + epsilon0) / (e_perc * self.n_epochs) * ep + epsilon1, epsilon0])

        logits = self.actor(observation_tensor.float())
        logits_softmax = self.actor.transform_with_softmax(logits)

        if epsilon >= np.random.random():
            action = logits_softmax.sample()

        else:
            dummy_dist = th.ones_like(logits_softmax.probs).numpy() / logits_softmax.probs.shape[2]
            action = th.Tensor(
                [[np.random.choice(range(dummy_dist.shape[2]), replace=True, p=dist) for dist in dummy_dist[0]]])
            action = action.int()

        value = th.dot(self.critic(observation_tensor.float()), th.from_numpy(np.asarray(self.env.w_rewards)).float())
        log_probs = th.squeeze(logits_softmax.log_prob(action))
        action = th.squeeze(action, dim=0).numpy()

        value = th.squeeze(value).item()
        # The probability of following the action vector is the sum of the log_probs of the probabilities of each action
        prob = log_probs.sum()

        return action, prob, value

    def run_episodes(self, exec_mode="train", checkpoint_dir=None, checkpoint_ep=None, reuse_mode="full"):
        # Iterate over episodes
        # If we are in training mode
        if (exec_mode == "train") or (exec_mode == "continue_training"):

            if exec_mode == "continue_training":
                self._load_model_weights(checkpoint_dir, checkpoint_ep, reuse_mode)

            if not self.quiet: print(f"Starting training.")
            for episode in range(self.n_epochs):
                returns = self.run_episode(episode, train_phase="learn")
                # log everything that is needed
                if episode % self.checkpoint_interval == 0 or episode == self.n_epochs - 1:
                    self.actor.save_checkpoint(episode)
                    self.critic.save_checkpoint(episode)

                values = np.dot(self.critic(th.Tensor(self.env.states_nn)).detach().numpy(),
                                self.env.w_rewards)
                self.log_after_train_episode(episode, returns, values)

                if self.test_interval:
                    if episode % self.test_interval == 0 or episode == self.n_epochs - 1:
                        if not self.quiet: print(f"Beginning test runs with current weights.")
                        test_rewards = []
                        for test_episode in range(self.test_n_epochs):
                            test_rewards.append(self.run_episode(test_episode, train_phase="test_train"))
                        self.total_rewards_test.append(np.mean(test_rewards))
                        self.log_after_test_episode(np.mean(test_rewards), episode)

        elif exec_mode == "test":
            self._load_model_weights(checkpoint_dir, checkpoint_ep, reuse_mode)

            if not self.quiet: print(f"Starting testing")
            for episode in range(self.test_n_epochs):
                self.run_episode(episode, train_phase="test")
        else:
            raise ValueError(f"Execution mode {exec_mode} is not supported. Available options are " \
                             "learn, test, continue_learning")

        # Save the retrieved results (actions, rewards)
        # np.save(self.output_dir + "actions.npy", self.total_actions)
        np.save(self.output_dir + "rewards.npy", self.total_rewards)
        np.save(self.output_dir + "actions_test.npy", self.total_actions_test)
        np.save(self.output_dir + "rewards_test.npy", self.total_rewards_test)

        self.clear_bad_checkpoints()

    def run_episode(self, episode, train_phase="learn"):
        # Initialize buffer
        self.buffer.reset_buffer()

        # Iterate over time steps
        cur_timestep = 0
        episode_returns = []
        # total_urgent_comps = 0
        while cur_timestep < self.env.timesteps:
            observation = self.env.states_nn
            # Sample action from actor, and value from critic
            action, log_prob, value = self.sample_action(self.env.states_nn, episode)

            # Perform a step into the environment
            observation_new, reward, done, _ = self.env.step(self.env.actions[copy(action)])
            # total_urgent_comps += len(self.env.urgent_comps)

            # Store the observation, action, reward, predicted value and log probabilities
            self.buffer.store(observation, action, reward, value, log_prob)

            # Check if the episode has ended. If so, proceed to training
            if done or (cur_timestep == self.env.timesteps - 1):
                observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float).to(
                    self.device)
                last_value = 0 if done else th.dot(self.critic(observation_tensor.reshape(1, -1)).item(), self.env.w_rewards)
                self.buffer.finish_trajectory(last_value, self.env.w_rewards)
                self.env.reset()

                # Check where to append the rewards based on the execution mode
                if train_phase == "learn" or train_phase == "test_train" or train_phase == "return_values":
                    pass
                    # self.total_rewards.append(np.sum(
                    #     self.buffer.reward_buffer) * self.env.norm_factor)
                    # # self.total_actions.append(self.buffer.action_buffer)
                elif train_phase == "test":
                    pass
                    self.total_rewards_test.append(
                        np.einsum("ij,j->ij", self.buffer.reward_buffer, self.env.norm_factor))
                    self.total_actions_test.append(self.buffer.action_buffer)
                else:
                    raise ValueError(
                        f"Execution mode {train_phase} not relevant. Available options are learn and test.")

                act, counts = np.unique(self.buffer.action_buffer, return_counts=True)
                if not self.quiet: print(f"{train_phase} episode: {episode}, Total return:"
                      # f" {np.sum(self.buffer.reward_buffer, axis=0) * self.env.norm_factor} "
                      f" {self.buffer.return_buffer[0] * self.env.norm_factor} "
                      f"Actions percentages {dict(zip(act.astype(int), counts * 100 // (self.env.num_components * self.env.timesteps)))}"
                      # f"Total urgent comps {total_urgent_comps}"
                      )
                break

            cur_timestep += 1

        # Train the two networks based on the experience of this episode
        if train_phase == "learn":
            actor_loss, critic_loss = self.train()

            self.log_after_training(episode, actor_loss, critic_loss)

        if train_phase == "return_values":
            return self.buffer.return_buffer[0] # TODO: Get mean of these as return of OLS!

        return np.sum(self.buffer.return_buffer[0] * self.env.w_rewards)

    def train(self):
        if self.normalize_advantage:
            self.buffer.advantage_buffer = self._normalize_array(self.buffer.advantage_buffer)

        if self.normalize_returns:
            self.buffer.return_buffer = self._normalize_array(self.buffer.return_buffer)

        # Get the complete buffer values
        (observation_buffer_init,
         action_buffer_init,
         advantage_buffer_init,
         return_buffer_init,
         logprobability_buffer_init) = self.buffer.get()

        continue_training = True

        for epoch in range(self.train_iters):

            permutation = th.randperm(self.env.timesteps * self.processes)

            for i in range(0, self.env.timesteps, self.batch_size):
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                indices = permutation[i:i + self.batch_size]

                observation_buffer = th.Tensor(observation_buffer_init[indices]).to(self.device)
                logprobability_buffer = th.Tensor(logprobability_buffer_init[indices]).to(
                    self.device)
                action_buffer = th.Tensor(action_buffer_init[indices]).to(self.device)
                return_buffer = th.Tensor(return_buffer_init[indices]).to(self.device)
                advantage_buffer = th.Tensor(advantage_buffer_init[indices]).to(self.device)

                critic_values = th.einsum("ij,j->i", self.critic(observation_buffer), th.from_numpy(np.asarray(self.env.w_rewards)).float())
                return_buffer = th.einsum("ij,j->i", return_buffer, th.from_numpy(np.asarray(self.env.w_rewards)).float())
                critic_values = th.squeeze(critic_values)
                return_buffer = th.squeeze(return_buffer)

                try:
                    state_dist = self.actor.transform_with_softmax(self.actor(observation_buffer))
                except:
                    print("Found the error!")
                new_probs = state_dist.log_prob(action_buffer).sum(dim=1)

                actor_loss = {"policy_loss": self._policy_loss(logprobability_buffer, new_probs, advantage_buffer)}
                critic_loss = self._value_loss(return_buffer, critic_values)

                # Add the entropy coefficient to the actor loss
                if self.ent_coef:
                    self.entropy = state_dist.entropy()
                    actor_loss["entropy_loss"] = -self.ent_coef * th.mean(self.entropy)

                if self.vf_coef:
                    actor_loss["value_loss"] = self.vf_coef * critic_loss.detach()

                actor_loss["total_loss"] = sum(actor_loss.values())
                actor_loss["total_loss"].backward()
                critic_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                with th.no_grad():
                    log_ratio = new_probs - logprobability_buffer
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if not self.env.quiet or True:
                        if not self.quiet: print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.4f}")
                        continue_training = False
                    break

            if not continue_training:
                break

        # Progress the learning rate schedulers by one
        self.actor.scheduler.step()
        self.critic.scheduler.step()

        return actor_loss, critic_loss.detach().numpy()

    @staticmethod
    def _normalize_tensor(arr):
        mean, std = (th.mean(arr), th.std(arr))

        return (arr - mean) / std

    def _normalize_advantages(self):
        self.buffer.counter, self.buffer.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (np.mean(self.buffer.advantage_buffer),
                                         np.std(self.buffer.advantage_buffer))

        self.buffer.advantage_buffer = (self.buffer.advantage_buffer - advantage_mean) / advantage_std

    @staticmethod
    def _normalize_array(arr):
        # self.buffer.counter, self.buffer.trajectory_start_index = 0, 0
        arr_mean, arr_std = (np.mean(arr), np.std(arr))

        if arr_std == 0:
            arr_std = 1

        return (arr - arr_mean) / arr_std

    def _policy_loss(self, old_pi, new_pi, advantages):
        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(new_pi - old_pi)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss

    def _value_loss(self, returns, values):

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = self.buffer.value_buffer + th.clamp(
                values - self.buffer.value_buffer, -self.clip_range_vf, self.clip_range_vf
            )

        value_loss = F.mse_loss(returns, values_pred)
        return value_loss

    def _load_model_weights(self, checkpoint_dir, checkpoint_ep, reuse_mode):
        if not self.quiet: print(f"Loading specified weights")
        if (not checkpoint_dir) or (not str(checkpoint_ep)):
            raise ValueError("In order to run in test mode, you need to specify a checkpoint "
                             "directory (checkpoint_dir) and a checkpoint epoch (checkpoint_ep)")
        self.actor.load_checkpoint(checkpoint_dir, checkpoint_ep, reuse_mode)
        self.critic.load_checkpoint(checkpoint_dir, checkpoint_ep, reuse_mode)

    def _load_yaml_file(self):
        """
        Method that loads the parameter file.
        :return: dictionary containing all parameters
        """
        with open(self.param_file, 'rb') as f:
            return yaml.safe_load(f.read())

    def _load_params(self, inp=None):
        """
        Loads all parameters from the parameter files to class attributes
        :param inp: The input to make an attribute
        :return: -
        """
        if not inp:
            inp = self.param_dict
        for k, v in inp.items():
            if isinstance(v, dict):
                self._load_params(inp=v)
            else:
                setattr(self, k, v)

    def log_after_train_episode(self, episode, returns, values):
        if self.log_enabled and episode % self.log_interval == 0:
            self.writer.add_scalars(self.log["train_returns"],
                                    {"return": returns,
                                     "value": values},
                                   # {"Lifecycle cost": returns[0],
                                   #  "Lifecycle carbon emissions": returns[1],
                                   #  "Total travel time": returns[2]},
                                   episode)

            self.writer.add_scalar(self.log["lr_actor"],
                                   self.actor.optimizer.param_groups[0]["lr"],
                                   episode)

            self.writer.add_scalar(self.log["lr_critic"],
                                   self.critic.optimizer.param_groups[0]["lr"],
                                   episode)
            # self.writer.add_scalar(self.log["critic_value"],
            #                        values,
            #                        episode)

    def log_after_test_episode(self, returns, episode):
        if self.log_enabled and episode % self.log_interval == 0:
            self.writer.add_scalar(self.log["test_returns"],
                                   returns,
                                   # {"Lifecycle cost": np.mean(test_rewards)[0],
                                   #  "Lifecycle carbon emissions": np.mean(test_rewards)[1],
                                   #  "Total travel time": np.mean(test_rewards)[2]},
                                   episode)

    def log_after_training(self, episode, actor_loss, critic_loss):
        if self.log_enabled and episode % self.log_interval == 0:
            self.writer.add_scalars(self.log["actor_loss"],
                                    {key: value.item() for key, value in actor_loss.items()},
                                    episode)
            self.writer.add_scalar(self.log["critic_loss"],
                                   critic_loss,
                                   episode)

    def log_at_start(self):
        self.writer.add_graph(self.actor, th.Tensor(self.env.states_nn))
        # self.writer.add_graph(self.actor)

        dict_to_log = {key: str(value) for key, value in self.param_dict.items()}
        self.writer.add_hparams(dict_to_log, {})


    def clear_bad_checkpoints(self):
        best_episode = np.clip(np.argmax(np.asarray(self.total_rewards_test))*self.test_interval,
                               a_min=None, a_max=self.n_epochs-1)

        checkpoint_episodes = np.arange(0, self.n_epochs, self.checkpoint_interval)
        checkpoint_episodes = np.append(checkpoint_episodes, self.n_epochs - 1)

        episode_to_keep1 = checkpoint_episodes[(np.abs(checkpoint_episodes - best_episode)).argmin()]
        episode_to_keep2 = self.n_epochs - 1

        # Remove all other episode weights
        for filee in os.listdir(self.actor.checkpoint_folder):
            if ("ep" + str(episode_to_keep1) + "_" not in filee) and ("ep" + str(episode_to_keep2) + "_" not in filee):
                os.remove(self.actor.checkpoint_folder + filee)

        self.best_result = np.max(self.total_rewards_test)
        self.best_weight = episode_to_keep1
        print(f"Best result: {self.best_result}")
        print(f"Value of best network weights: {best_episode:.0f}")

if __name__ == "__main__":
    import time

    start = time.time()
    os.chdir("../")

    # f = open("ppo_results_2_comps.txt", "w+")
    # for checkpoint in range(0, 20000, 250):
    #     f.write(f"Checking weights {checkpoint}\n")
    ppo = MindmapPPO()
    ppo.run_episodes(exec_mode="train")
    # ppo.run_episodes(exec_mode="test", checkpoint_dir="src/model_weights/20230222162100_0290/",
    #                  checkpoint_ep=18500)
    # ppo.run_episodes(exec_mode="continue_training",checkpoint_dir="src/model_weights/20230228181434_311/",
    #                  checkpoint_ep=199)
    # print(f"Mean of test rewards: {np.mean(ppo.total_rewards_test)}\n")
    print(f"Total time {time.time() - start}")
