import time

import torch as th
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy, copy
import numbers
from datetime import datetime
# Multi-threading
import threading

import matplotlib.pyplot as plt
import pylab as pl
# Reinforcement learning
import gym
# import cv2

# ! pip install openmatrix

import gym
import thesis_env
import os
import math
import shutil
import yaml
from torch import nn
from scipy.signal import lfilter
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F


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

    def __init__(self, num_states, num_actions, num_components, timesteps, num_objectives, gamma=0.95, lam=0.95, processes=1):
        self.processes = processes  # TODO Make it a parameter

        # Buffer initialization
        self.num_components = num_components
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # The observation dimension equals to components*(states_num)
        # TODO - IRI is stationary, so state embedding only consists of IRI
        self.observation_dimensions = self.num_components * (self.num_states)
        # self.observation_dimensions = self.num_components * (self.num_states + 1)
        self.size = timesteps * self.processes
        self.gamma = gamma
        self.lam = lam

        self.lock = threading.Lock()

        # Reset buffer
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.observation_buffer = np.zeros((self.size, self.observation_dimensions),
                                           dtype=np.float32)
        self.action_buffer = np.zeros((self.size, self.num_components), dtype=np.float32)
        self.advantage_buffer = np.zeros(self.size, dtype=np.float32)
        self.reward_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
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
        self.lock.acquire()
        try:
            self.observation_buffer[self.counter] = observation
            self.action_buffer[self.counter] = action
            self.reward_buffer[self.counter] = reward
            self.value_buffer[self.counter] = value
            self.logprobability_buffer[self.counter] = logprobability.detach()
            self.counter += 1
        finally:
            self.lock.release()

    def finish_trajectory(self, last_value=0, w_rewards=None):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.counter)

        rewards = np.append(self.reward_buffer[path_slice], [[last_value] * self.num_objectives], axis=0)
        rewards_dot = np.einsum('ij,j->i', rewards, w_rewards)

        values = np.append(self.value_buffer[path_slice], last_value)
        deltas = (rewards_dot + self.gamma * np.roll(values, -1) - values)[:-1]
        # self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(rewards_dot, self.gamma)[:-1]

        # TODO - Simplification
        self.advantage_buffer[path_slice] = self.return_buffer[path_slice] - values[:-1]

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
        # This is equivalent
        # summ = [x[-1]]
        # for val in reversed(x[:-1]):
        #     summ.append(summ[-1] * discount + val)
        # return summ[::-1]


# Same as in exercise sheet 3
class Runner:
    """ Implements a simple single-thread runner class. """

    def __init__(self, controller, exploration_step=1):
        self.env = gym.make('thesis-env-v1', quiet=True)
        self.env.reset()
        self.controller = controller
        # Set up current state and time step
        self.sum_rewards = 0
        self.state = None
        self.time = 0

    #         self._next_step()

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()

    def transition_format(self):
        """ Returns the format of transtions: a dictionary of (shape, dtype) entries for each key. """
        return {'actions': ((1,), th.long),
                'states': (self.state_shape, th.float32),
                'next_states': (self.state_shape, th.float32),
                'rewards': ((1,), th.float32),
                'dones': ((1,), th.bool),
                'returns': ((1,), th.float32)}

    def _wrap_transition(self, s, a, r, ns, d):
        """ Takes a transition and returns a corresponding dictionary. """
        trans = {}
        form = self.transition_format()
        for key, val in [('states', s), ('actions', a), ('rewards', r), ('next_states', ns), ('dones', d)]:
            if not isinstance(val, th.Tensor):
                if isinstance(val, numbers.Number) or isinstance(val, bool): val = [val]
                val = th.tensor(val, dtype=form[key][1])
            if len(val.shape) < len(form[key][0]) + 1: val = val.unsqueeze(dim=0)
            trans[key] = val
        return trans

    def _run_step(self, a):
        """ Make a step in the environment (and update internal bookeeping) """
        ns, r, d, _ = self.env.step(a.item())
        self.sum_rewards += r
        return r, ns, d

    def _next_step(self, done=True, next_state=None):
        """ Switch to the next time-step (and update internal bookeeping) """
        self.time = 0 if done else self.time + 1
        if done:
            self.sum_rewards = 0
            self.state = self.env.reset()
            if self.use_pixels: self.state = self._pixel_observation(reset=True)
        else:
            self.state = next_state

    def run(self, n_steps, blueprint, transition_buffer_dict=None):  # TODO add transition_buffer + episode):
        # Initialize buffer for this runner
        my_transition_buffer = copy(blueprint)
        my_transition_buffer.reset_buffer()

        # Iterate over time steps
        cur_timestep = 0
        # total_urgent_comps = 0
        total_rewards = 0
        while cur_timestep < self.env.timesteps:
            observation = self.env.states_nn
            # Sample action from actor, and value from critic
            action, log_prob, value = self.controller.sample_action(self.env.states_nn, ep=0)

            # Perform a step into the environment

            observation_new, reward, done, _ = self.env.step(self.env.actions[copy(action)])
            total_rewards += reward
            # total_urgent_comps += len(self.env.urgent_comps)

            # Store the observation, action, reward, predicted value and log probabilities
            my_transition_buffer.store(observation, action, reward, value, log_prob)

            # Check if the episode has ended. If so, proceed to training
            if done or (cur_timestep == self.env.timesteps - 1):
                observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float)
                last_value = 0 if done else self.critic(observation_tensor.reshape(1, -1)).item()
                my_transition_buffer.finish_trajectory(last_value, self.env.w_rewards)
                self.env.reset()

                # Check where to append the rewards based on the execution mode
                #                 if train_phase == "learn":
                #                     pass
                #                     # self.total_rewards.append(np.sum(
                #                     #     self.buffer.reward_buffer) * self.env.norm_factor)
                #                     # # self.total_actions.append(self.buffer.action_buffer)
                #                 elif train_phase == "test":
                #                     pass
                #                     # self.total_rewards_test.append(np.sum(self.buffer.reward_buffer) * self.env.norm_factor)
                #                     # self.total_actions_test.append(self.buffer.action_buffer)
                #                 else:
                #                     raise ValueError(
                #                         "Execution mode {train_phase} not relevant. Available options are " \
                #                         "learn and test.")

                act, counts = np.unique(my_transition_buffer.action_buffer[:self.env.timesteps], return_counts=True)
                print(
                    #                     f"{train_phase} episode: {episode}, Total return:"
                    f"{np.sum(my_transition_buffer.reward_buffer, axis=0) * self.env.norm_factor} "
                    # f" {my_transition_buffer.reward_buffer.sum() * self.env.norm_factor[0]} "
                    f"Actions percentages {dict(zip(act.astype(int), counts * 100 // (self.env.num_components * my_transition_buffer.counter)))}"
                    # f"Total urgent comps {total_urgent_comps}"
                )

                if not transition_buffer_dict:
                    transition_buffer = my_transition_buffer
                else:
                    transition_buffer = transition_buffer_dict["transition_buffer"]

                    cur_count = transition_buffer.counter

                    transition_buffer.observation_buffer[cur_count:cur_count + 20] = my_transition_buffer.observation_buffer[
                                                                                cur_count:cur_count + 20]

                    transition_buffer.action_buffer[cur_count:cur_count + 20] = my_transition_buffer.action_buffer[
                                                                                cur_count:cur_count + 20]
                    transition_buffer.advantage_buffer[cur_count:cur_count + 20] = my_transition_buffer.advantage_buffer[
                                                                                   cur_count:cur_count + 20]
                    transition_buffer.reward_buffer[cur_count:cur_count + 20] = my_transition_buffer.reward_buffer[
                                                                                cur_count:cur_count + 20]
                    transition_buffer.return_buffer[cur_count:cur_count + 20] = my_transition_buffer.return_buffer[
                                                                                cur_count:cur_count + 20]
                    transition_buffer.value_buffer[cur_count:cur_count + 20] = my_transition_buffer.value_buffer[
                                                                               cur_count:cur_count + 20]
                    transition_buffer.logprobability_buffer[
                    cur_count:cur_count + 20] = my_transition_buffer.logprobability_buffer[cur_count:cur_count + 20]
                    transition_buffer.counter += 20
                break

            cur_timestep += 1

        transition_buffer_dict.update({"transition_buffer": transition_buffer})
        return transition_buffer_dict

    def run_episode(self, transition_buffer=None, trim=True, return_dict=None):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        return self.run(0, transition_buffer)

class MultiRunner:
    """ Simple class that runs multiple Runner objects in parallel and merges their outputs. """

    def __init__(self, controller, processes):
        self.workers = []
        self.runners = []
        n = processes
        for _ in range(n):
            self.runners.append(Runner(controller=controller))

    def transition_format(self):
        """ Same transition-format as underlying Runners. """
        return self.runners[0].transition_format()

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        # Join all workers
        for w in self.workers:
            w.join()
        # Exit all environments
        for r in self.runners:
            r.close()

    def fork(self, target, common_args=None, specific_args=None):
        """ Executes the function "target" on all runners. "common_args" is a dictionary of
            arguments that are passed to all runners, "specific_args" is a list of
            dictionaries that contain individual parameters for each runner. """
        # Fork all runners
        self.workers = []
        for i, r in enumerate(self.runners):
            r_args = [] if specific_args is None else [arg[i] for arg in specific_args]
            self.workers.append(threading.Thread(target=target, args=(r, *common_args, *r_args)))
            self.workers[-1].start()
        # Join all runners
        for w in self.workers:
            w.join()

    def run(self, n_steps, blueprint, transition_buffer_dict):
        """ Runs n_steps, split amongst runners, and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        n_steps = n_steps // len(self.runners)
        transition_buffer_dicts = [{} for _ in self.runners]
        self.fork(target=Runner.run, common_args=(n_steps, blueprint), specific_args=(transition_buffer_dicts,))
        # self.fork(target=Runner.run, common_args=(n_steps, transition_buffer))
        return transition_buffer_dicts

    def run_episode(self, transition_buffer=None, trim=True):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        return self.run(0, transition_buffer, trim)

class ACController:
    """ Controller for Q-value functions, synchronizes the model calls. """

    def __init__(self, num_actions, actor, critic, n_epochs, model=None):
        self.lock = threading.Lock()
        self.num_actions = 5  # TODO, update
        if model:
            self.model = model

        self.n_epochs = n_epochs
        self.actor = actor
        self.critic = critic

    def copy(self):
        """ Shallow copy of this controller that does not copy the model. """
        return QController(model=self.model, num_actions=self.num_actions)

    def parameters(self):
        """ Returns a generator of the underlying model parameters. """
        return self.model.parameters()

    def sanitize_inputs(self, observation, **kwargs):
        """ Casts numpy arrays as Tensors. """
        if isinstance(observation, np.ndarray):
            observation = th.Tensor(observation).unsqueeze(dim=0)
        return observation

    def probabilities(self, observation, precomputed=False, **kwargs):
        self.lock.acquire()
        try:
            mx = observation if precomputed else self.model(self.sanitize_inputs(observation))[:, :self.num_actions]
        finally:
            self.lock.release()
        return th.nn.functional.softmax(mx, dim=-1)

    def sample_action(self, observation, ep=0):
        observation_tensor = th.tensor(np.array(observation), dtype=th.float)

        epsilon0 = .0
        epsilon1 = .0
        e_perc = 0.3
        epsilon = np.max([(- epsilon1 + epsilon0) / (e_perc * self.n_epochs) * ep + epsilon1, epsilon0])
        # print(epsilon)
        # epsilon = np.max(
        #     (np.min((epsilon0 * (1 - ep / (.001 * self.n_epochs)) + epsilon1 * (ep / (.001 * self.n_epochs)),
        #              epsilon1)),
        #      0))

        self.lock.acquire()
        logits = self.actor(observation_tensor.float())
        self.lock.release()
        logits_softmax = self.actor.transform_with_softmax(logits)

        # if epsilon > np.random.random():
        if True:

            action = logits_softmax.sample()

        else:
            dummy_dist = th.ones_like(logits_softmax.probs).numpy() / logits_softmax.probs.shape[2]
            action = th.Tensor(
                [[np.random.choice(range(dummy_dist.shape[2]), replace=True, p=dist) for dist in dummy_dist[0]]])
            action = action.int()

        value = self.critic(observation_tensor.float())
        log_probs = th.squeeze(logits_softmax.log_prob(action))
        action = th.squeeze(action, dim=0).numpy()

        value = th.squeeze(value).item()

        # the probability of following the action vector is the sum of the log_probs of the
        # probabilities of each action
        prob = log_probs.sum()

        return action, prob, value

class MindmapPPO:
    """
    The main class containing the MindmapPPO algorithm
    """

    def __init__(self, param_file="src/model_params_mt.yaml", quiet=False):

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

        if not self.multirunner:
            self.processes = 1

        # Initialize Rollout buffer
        self.buffer = MindmapRolloutBuffer(self.env.num_states_iri, self.env.num_actions,
                                           self.env.num_components, self.env.timesteps, self.env.num_objectives,
                                           self.gamma, self.lam, self.processes)

        # Initialize actor and critic networks
        self.actor = MindmapActor(self.env.num_components, self.env.num_states_iri,
                                  self.env.num_actions, self.device, self.timestamp,
                                  self.optimizer_act, self.lr_act, self.lr_act_min,
                                  self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                  self.actor_arch, self.checkpoint_dir)
        self.critic = MindmapCritic(self.env.num_components, self.env.num_states_iri,
                                    self.env.num_actions, self.device, self.timestamp,
                                    self.optimizer_crit, self.lr_crit, self.lr_crit_min,
                                    self.lr_decay_step, self.lr_decay_episode_perc, self.n_epochs,
                                    self.critic_arch, self.checkpoint_dir, checkpoint_suffix="crit")

        # Multi-threading execution

        self.controller = ACController(self.env.num_actions, self.actor, self.critic, self.n_epochs)
        self.runner = MultiRunner(self.controller, processes=self.processes) if (self.multirunner == True) \
            else Runner(self.controller)

        self.set_controller(self.controller)

    def set_controller(self, controller):
        """ This function is called in the experiment to set the controller. """
        self.controller = controller

    def run_episodes(self, exec_mode="train", checkpoint_dir=None, checkpoint_ep=None):
        # Iterate over episodes
        # If we are in training mode
        if (exec_mode == "train") or (exec_mode == "continue_training"):

            if exec_mode == "continue_training":
                self._load_model_weights(checkpoint_dir, checkpoint_ep)

            print(f"Starting training.")
            for episode in range(self.n_epochs):
                print(f"Episode {episode}:")
                self.buffer.reset_buffer()
                transition_buffers_list = self.runner.run(self.env.timesteps, blueprint=self.buffer, transition_buffer_dict={"transition_buffer": self.buffer})

                if isinstance(transition_buffers_list, dict):
                    transition_buffers_list = [transition_buffers_list]

                buff_count = 0
                for buffer in transition_buffers_list:
                    buff = buffer["transition_buffer"]

                    self.buffer.observation_buffer[buff_count:buff_count + self.env.timesteps] = buff.observation_buffer[:self.env.timesteps]
                    self.buffer.action_buffer[buff_count:buff_count + self.env.timesteps] = buff.action_buffer[:self.env.timesteps]
                    self.buffer.advantage_buffer[buff_count:buff_count + self.env.timesteps] = buff.advantage_buffer[:self.env.timesteps]
                    self.buffer.reward_buffer[buff_count:buff_count + self.env.timesteps] = buff.reward_buffer[:self.env.timesteps]
                    self.buffer.return_buffer[buff_count:buff_count + self.env.timesteps] = buff.return_buffer[:self.env.timesteps]
                    self.buffer.value_buffer[buff_count:buff_count + self.env.timesteps] = buff.value_buffer[:self.env.timesteps]
                    self.buffer.logprobability_buffer[buff_count:buff_count + self.env.timesteps] = buff.logprobability_buffer[:self.env.timesteps]
                    buff_count += self.env.timesteps

                #                 _ = self.run_episode(episode, train_phase="learn")
                # log everything that is needed
                if episode % self.checkpoint_interval == 0 or episode == self.n_epochs - 1:
                    self.actor.save_checkpoint(episode)
                    self.critic.save_checkpoint(episode)

                # Train the two networks based on the experience of this episode
                self.train()

                if self.test_interval:
                    if episode % self.test_interval == 0 or episode == self.n_epochs - 1:
                        print(f"Beginning test runs with current weights.")
                        test_rewards = []
                        for test_episode in range(self.test_n_epochs):
                            test_rewards.append(self.run_episode(test_episode, train_phase="test_train"))
                        if exec_mode != "test":
                            self.total_rewards_test.append(np.mean(test_rewards))


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
        # ONLY USED IN TESTING!
        # Initialize buffer
        self.buffer.reset_buffer()

        # Iterate over time steps
        cur_timestep = 0
        episode_returns = []
        # total_urgent_comps = 0
        while cur_timestep < self.env.timesteps:
            observation = self.env.states_nn
            # Sample action from actor, and value from critic
            action, log_prob, value = self.controller.sample_action(self.env.states_nn, episode)

            # Perform a step into the environment

            observation_new, reward, done, _ = self.env.step(self.env.actions[copy(action)])
            # total_urgent_comps += len(self.env.urgent_comps)

            # Store the observation, action, reward, predicted value and log probabilities
            self.buffer.store(observation, action, reward, value, log_prob)

            # Check if the episode has ended. If so, proceed to training
            if done or (cur_timestep == self.env.timesteps - 1):
                observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float).to(
                    self.device)
                last_value = 0 if done else self.critic(observation_tensor.reshape(1, -1)).item()
                self.buffer.finish_trajectory(last_value, self.env.w_rewards)
                self.env.reset()

                # Check where to append the rewards based on the execution mode
                if train_phase == "learn" or train_phase == "test_train":
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

                act, counts = np.unique(self.buffer.action_buffer[:self.env.timesteps], return_counts=True)
                print(f"{train_phase} episode: {episode}, Total return:"
                      f" {np.sum(self.buffer.reward_buffer, axis=0) * self.env.norm_factor} "
                      f"Actions percentages {dict(zip(act.astype(int), counts * 100 // (self.env.num_components * self.env.timesteps)))}"
                      # f"Total urgent comps {total_urgent_comps}"
                      )
                break

            cur_timestep += 1

        # Train the two networks based on the experience of this episode
        if train_phase == "learn":
            self.train()

        return np.sum(np.einsum('ij,j->i', self.buffer.reward_buffer, self.env.w_rewards * self.env.norm_factor))

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

                if self.vf_coef:
                    actor_loss += self.vf_coef * critic_loss.detach()

                # print(f"actor: {actor_loss-self.vf_coef * critic_loss.detach()+self.ent_coef * th.mean(self.entropy)}, vf: {self.vf_coef * critic_loss.detach()} entropy: {self.ent_coef * th.mean(self.entropy)}")

                actor_loss.backward()
                critic_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                with th.no_grad():
                    log_ratio = new_probs - logprobability_buffer
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if not self.env.quiet:
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

    @staticmethod
    def _normalize_tensor(arr):
        mean, std = (th.mean(arr), th.std(arr))

        return (arr - mean) / std

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
        # returns = self._normalize_tensor(returns)
        # values = self._normalize_tensor(values)

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
        # value_loss = F.mse_loss(returns[:-1], values_pred[:-1])
        value_loss = F.mse_loss(returns, values_pred)
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
        # TODO - Since IRI is stationary, we don't need to add the time in the state embedding
        self.input_dim = (self.num_states) * self.num_components
        # self.input_dim = (self.num_states + 1) * self.num_components
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

if __name__ == "__main__":
    os.chdir("../")

    start = time.time()
    ppo = MindmapPPO()
    print(f"Weights are {ppo.env.w_rewards}")
    ppo.run_episodes(exec_mode="train")
    print(f"Total time {time.time() - start} seconds.")