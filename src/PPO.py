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

from datetime import datetime


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

    def __init__(self, num_states, num_actions, num_components, timesteps, gamma=0.95, lam=0.95):
        # Buffer initialization
        self.num_components = num_components
        self.num_states = num_states
        self.num_actions = num_actions
        # The observation dimension equals to components*(states_num + deterioration rate)
        self.observation_dimensions = self.num_components * (self.num_states + 1)
        self.size = timesteps
        self.gamma = gamma
        self.lam = lam

        # Reset buffer
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.observation_buffer = np.zeros((self.size, self.observation_dimensions),
                                           dtype=np.float32)
        self.action_buffer = np.zeros((self.size, self.num_components), dtype=np.float32)
        self.advantage_buffer = np.zeros(self.size, dtype=np.float32)
        self.reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.return_buffer = np.zeros(self.size, dtype=np.float32)
        self.value_buffer = np.zeros(self.size, dtype=np.float32)
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

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.counter)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        deltas = (rewards + self.gamma * np.roll(values, -1) - values)[:-1]
        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam)
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma)[:-1]
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


class MindmapActor(nn.Module):
    def __init__(self, num_components, num_states, num_actions, device, timestamp, optimizer, lr,
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
        self.n_epochs = n_epochs
        self.device = device

        # Setup network input and output
        self.input_dim = (self.num_states + 1) * self.num_components
        if self.checkpoint_suffix == "actor":
            self.output_dim = self.num_components * self.num_actions
        else:
            self.output_dim = 1

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
                    self.layers.append(self.supported_activation_fns[layer])
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step,
                                                   gamma=self.gamma_decay)

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

    def load_checkpoint(self, checkpoint_dir, checkpoint_ep):
        checkpoint_path = f"{checkpoint_dir}ep{checkpoint_ep}_{self.checkpoint_suffix}.pth"
        self.load_state_dict(th.load(checkpoint_path, map_location=self.device))


class MindmapCritic(MindmapActor):
    def __init__(self, num_components, num_states, num_actions, device, timestamp, optimizer, lr,
                 lr_min, lr_decay_step, lr_decay_episode_perc, n_epochs, architecture,
                 checkpoint_dir,
                 checkpoint_suffix="critic"):
        super(MindmapCritic, self).__init__(num_components, num_states, num_actions,
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
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
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

        # [DEMO] Total number of states equals to states_cci + states_iri
        self.env.num_states = self.env.num_states_cci + self.env.num_states_iri

        # Initialize Rollout buffer
        self.buffer = MindmapRolloutBuffer(self.env.num_states, self.env.num_actions,
                                           self.env.num_components,
                                           self.env.timesteps, self.gamma, self.lam)

        # Initialize actor and critic networks
        self.actor = MindmapActor(self.env.num_components, self.env.num_states,
                                  self.env.num_actions, self.device, self.timestamp,
                                  self.optimizer_act, self.lr_act, self.lr_act_min,
                                  self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                  self.actor_arch, self.checkpoint_dir)
        self.critic = MindmapCritic(self.env.num_components, self.env.num_states,
                                    self.env.num_actions, self.device, self.timestamp,
                                    self.optimizer_crit, self.lr_crit, self.lr_crit_min,
                                    self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                    self.critic_arch, self.checkpoint_dir, checkpoint_suffix="crit")

    def sample_action(self, observation):
        observation_tensor = th.tensor(np.array([observation]), dtype=th.float).to(self.device)
        logits = self.actor(observation_tensor)
        logits_softmax = self.actor.transform_with_softmax(logits)
        action = logits_softmax.sample()

        value = self.critic(observation_tensor)
        log_probs = th.squeeze(logits_softmax.log_prob(action))
        action = th.squeeze(action, dim=0).numpy()

        value = th.squeeze(value).item()

        # the probability of following the action vector is the sum of the log_probs of the
        # probabilities of each action
        prob = log_probs.sum()

        return action, prob, value

    def run_episodes(self, exec_mode="train", checkpoint_dir=None, checkpoint_ep=None):
        # Iterate over episodes
        # If we are in training mode
        if (exec_mode == "train") or (exec_mode == "continue_training"):

            if exec_mode == "continue_training":
                self._load_model_weights(checkpoint_dir, checkpoint_ep)

            print(f"Starting training.")
            for episode in range(self.n_epochs):
                self.run_episode(episode, train_phase="learn")
                # TODO - log everything that is needed
                if episode % self.checkpoint_interval == 0 or episode == self.n_epochs - 1:
                    self.actor.save_checkpoint(episode)
                    self.critic.save_checkpoint(episode)
                if self.test_interval:
                    if episode % self.test_interval == 0 or episode == self.n_epochs - 1:
                        print(f"Beginning test runs with current weights.")
                        for test_episode in range(self.test_n_epochs):
                            self.run_episode(test_episode, train_phase="test")

        elif exec_mode == "test":
            self._load_model_weights(checkpoint_dir, checkpoint_ep)

            print(f"Starting testing")
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

    def run_episode(self, episode, train_phase="learn"):
        # Initialize buffer
        self.buffer.reset_buffer()

        # Iterate over time steps
        cur_timestep = 0
        while cur_timestep < self.env.timesteps:
            observation = self.env.states_nn
            # Sample action from actor, and value from critic
            action, log_prob, value = self.sample_action(self.env.states_nn)

            # Perform a step into the environment
            observation_new, reward, done, _ = self.env.step(self.env.actions[action])

            # Store the observation, action, reward, predicted value and log probabilities
            self.buffer.store(observation, action, reward, value, log_prob)
            # self.buffer.store(observation, action, reward, value, log_prob)
            observation_new = observation

            # Check if the episode has ended. If so, proceed to training
            if done or (cur_timestep == self.env.timesteps - 1):
                observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float).to(
                    self.device)
                last_value = 0 if done else self.critic(observation_tensor.reshape(1, -1)).item()
                self.buffer.finish_trajectory(last_value)
                self.env.reset()

                # Check where to append the rewards based on the execution mode
                if train_phase == "learn":
                    self.total_rewards.append(np.sum(
                        self.buffer.reward_buffer) * self.env.norm_factor)
                    # self.total_actions.append(self.buffer.action_buffer)
                elif train_phase == "test":
                    self.total_rewards_test.append(np.sum(
                        self.buffer.reward_buffer) * self.env.norm_factor)
                    self.total_actions_test.append(self.buffer.action_buffer)
                else:
                    raise ValueError(
                        "Execution mode {train_phase} not relevant. Available options are " \
                        "learn and test.")

                print(f"{train_phase} episode: {episode}, Total return:"
                      f" {self.buffer.reward_buffer.sum() * self.env.norm_factor[0]}, and total "
                      f"actions: {self.env.actions[self.buffer.action_buffer[:, 0].astype(int)]}")
                break

            cur_timestep += 1

        # Train the two networks based on the experience of this episode
        if train_phase == "learn":
            self.train()

    def train(self):
        if self.normalize_advantage:
            self._normalize_advantages()

        # Get the complete buffer values
        (observation_buffer_init,
         action_buffer_init,
         advantage_buffer_init,
         return_buffer_init,
         logprobability_buffer_init) = self.buffer.get()

        continue_training = True

        for epoch in range(self.train_iters):

            permutation = th.randperm(self.env.timesteps)

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

                critic_values = self.critic(observation_buffer)
                critic_values = th.squeeze(critic_values)

                state_dist = self.actor.transform_with_softmax(self.actor(observation_buffer))
                new_probs = state_dist.log_prob(action_buffer).sum(dim=1)

                actor_loss = self._policy_loss(logprobability_buffer, new_probs, advantage_buffer)
                critic_loss = self._value_loss(return_buffer, critic_values)

                # Add the entropy coefficient to the actor loss
                if self.ent_coef:
                    self.entropy = state_dist.entropy()
                    actor_loss -= self.ent_coef * th.mean(self.entropy)

                actor_loss.backward()
                critic_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                with th.no_grad():
                    log_ratio = new_probs - logprobability_buffer
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if not self.env.quiet or True:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.4f}")
                        continue_training = False
                    break

            if not continue_training:
                break

        # Progress the learning rate schedulers by one
        self.actor.scheduler.step()
        self.critic.scheduler.step()

    def _normalize_advantages(self):
        self.buffer.counter, self.buffer.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (np.mean(self.buffer.advantage_buffer),
                                         np.std(self.buffer.advantage_buffer))

        self.buffer.advantage_buffer = (self.buffer.advantage_buffer - advantage_mean) / advantage_std

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
            # Value loss using the TD(gae_lambda) target
            # TODO: Temporal check: not take into account the last value
        value_loss = F.mse_loss(returns[:-1], values_pred[:-1])
        return value_loss

    def _load_model_weights(self, checkpoint_dir, checkpoint_ep):
        print(f"Loading specified weights")
        if (not checkpoint_dir) or (not str(checkpoint_ep)):
            raise ValueError("In order to run in test mode, you need to specify a checkpoint "
                             "directory (checkpoint_dir) and a checkpoint epoch (checkpoint_ep)")
        self.actor.load_checkpoint(checkpoint_dir, checkpoint_ep)
        self.critic.load_checkpoint(checkpoint_dir, checkpoint_ep)

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


if __name__ == "__main__":
    os.chdir("../")

    ppo = MindmapPPO()
    ppo.run_episodes(exec_mode="train")
    # ppo.run_episodes(exec_mode="test", checkpoint_dir="src/model_weights/20221102221133/",
    #                  checkpoint_ep=19750)
    # ppo.run_episodes(exec_mode="continue_training",checkpoint_dir="src/model_weights/20221003215042/",
    #                  checkpoint_ep=1750)
    print(f"Mean of test rewards: {np.mean(ppo.total_rewards_test)}")
