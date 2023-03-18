import os
from copy import copy, deepcopy
import time
import torch as th
import numpy as np
import gym
from scipy.signal import lfilter

import torch.multiprocessing as mp
from collections import namedtuple
mp.set_start_method("spawn", force=True)

import thesis_env
from PPO import MindmapPPO

th.set_num_interop_threads(1) # TODO - take a look at those
th.set_num_threads(1)

# creating msgs for communication between subprocess and main process.
# for when agent reached logging episode
MsgRewardInfo = namedtuple('MsgRewardInfo', ['agent', 'episode', 'metadata'])
# for when agent reached update timestep
MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'episode', 'buffer', 'update'])
# for when agent reached max episodes
MsgMaxReached = namedtuple('MsgMaxReached', ['agent', 'reached'])
# instruct the agent whether the current run is in train or test mode
MsgTrainMode = namedtuple('MsgTrainMode', ['train'])

class MindmapRolloutBufferMultithread:
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
    def __init__(self, num_states, num_components, timesteps, num_objectives, gamma=0.95, lam=0.95, processes=1, device="cpu"):
        # Buffer initialization
        self.num_components = num_components
        self.num_states = num_states
        self.num_objectives = num_objectives
        # The observation dimension equals to components*(states_num)
        # IRI is stationary, so state embedding only consists of IRI
        self.observation_dimensions = self.num_components * (self.num_states) + 1
        self.gamma = gamma
        self.lam = lam
        self.device = device

        # Parameters related to multiprocessing
        self.processes = processes
        self.size = timesteps * self.processes

        # Reset buffer
        self.reset_buffer()

    def init_shared_tensor(self, dims=None):
        return th.zeros(dims).to(self.device).share_memory_()

    def reset_buffer(self) -> None:
        self.observation_buffer = self.init_shared_tensor((self.size, self.observation_dimensions))
        self.action_buffer = self.init_shared_tensor((self.size, self.num_components))
        self.advantage_buffer = self.init_shared_tensor(self.size).to(self.device)
        self.reward_buffer = self.init_shared_tensor((self.size, self.num_objectives))
        self.return_buffer = self.init_shared_tensor((self.size, self.num_objectives))
        self.value_buffer = self.init_shared_tensor((self.size, self.num_objectives))
        self.logprobability_buffer = self.init_shared_tensor(self.size).to(self.device)
        self.counter, self.trajectory_start_index = 0, 0

    def array_to_tensor(self, arr):
        return th.tensor(arr).float()

    def store(self, observation, action, reward, value, logprobability, start_idx=None):
        """
        Append one step of agent-environment interaction
        :param observation: ndarray - Believe matrix (num_components x (num_states + 1))
        :param action: ndarray - Actions for the specific timestep (num_components)
        :param reward: float - Reward for the specific timestep
        :param value: float - Value for the specific timestep
        :param logprobability: ndarray - Log probabilities for selected actions (num_components x num_states)
        """
        if not start_idx:
            start_idx = self.counter

        self.observation_buffer[start_idx] = self.array_to_tensor(observation)
        self.action_buffer[start_idx] = self.array_to_tensor(action)
        self.reward_buffer[start_idx] = self.array_to_tensor(reward)
        self.value_buffer[start_idx] = self.array_to_tensor(value)
        self.logprobability_buffer[start_idx] = logprobability.detach()

        self.counter += 1

    def store_all(self, observation, action, advantage, reward, returns, value, logprobability, start_idx, end_idx):
        """
        Append one step of agent-environment interaction
        :param observation: tensor - Believe matrix (num_components x (num_states + 1))
        :param action: tensor - Actions for the specific timestep (num_components)
        :param advantage: tensor - Tensor with the advantage trajectory
        :param reward: tensor - Reward for the specific timestep
        :param returns: tensor - Tensor with the returns trajectory
        :param value: tensor - Value for the specific timestep
        :param logprobability: tensor - Log probabilities for selected actions (num_components x num_states)
        :param start_idx: int - the start index to place the trajectory
        :param end_idx: int - the end index to place the trajectory
        """

        self.observation_buffer[start_idx:end_idx] = observation.detach()
        self.action_buffer[start_idx:end_idx] = action.detach()
        self.advantage_buffer[start_idx:end_idx] = advantage.detach()
        self.reward_buffer[start_idx:end_idx] = reward.detach()
        self.return_buffer[start_idx:end_idx] = returns.detach()
        self.value_buffer[start_idx:end_idx] = value.detach()
        self.logprobability_buffer[start_idx:end_idx] = logprobability.detach()

        self.counter = end_idx

    def finish_trajectory(self, last_value=0, w_rewards=None):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.counter)
        rewards = np.append(self.reward_buffer.detach().numpy()[path_slice], [[last_value] * self.num_objectives], axis=0)

        values = np.append(self.value_buffer.detach().numpy()[path_slice], [[last_value] * self.num_objectives], axis=0)
        values_dot = np.einsum('ij,j->i', values, w_rewards)

        self.return_buffer[path_slice] = th.from_numpy(self.discounted_cumulative_sums(rewards, self.gamma)[:-1].copy())
        returns_dot = np.einsum('ij,j->i', self.return_buffer.detach().numpy()[path_slice], w_rewards)

        # Compute the advantage
        self.advantage_buffer[path_slice] = th.from_numpy(returns_dot - values_dot[:-1])

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


# class MindmapRolloutBufferMultithread_no_mem_share:
#     """
#     Buffer that stores the trajectory of the current episode. After the end of the episode it
#     calculates the advantage.
#
#     :param num_states (int): Number of discrete states per component
#     :param num_actions (int): Number of environmental actions
#     :param num_components (int): Number of discrete components
#     :param timesteps (int): Number of timesteps to in the environment
#     :param gamma (float): Discount factor
#     :param lam (float): Lambda parameter
#
#     """
#     def __init__(self, num_states, num_components, timesteps, num_objectives, gamma=0.95, lam=0.95, processes=1, device="cpu"):
#         # Buffer initialization
#         self.num_components = num_components
#         self.num_states = num_states
#         self.num_objectives = num_objectives
#         # The observation dimension equals to components*(states_num)
#         # IRI is stationary, so state embedding only consists of IRI
#         self.observation_dimensions = self.num_components * (self.num_states) + 1
#
#         self.processes = processes
#         self.size = timesteps * self.processes
#         self.gamma = gamma
#         self.lam = lam
#
#         # Reset buffer
#         self.reset_buffer()
#
#     def reset_buffer(self) -> None:
#         self.observation_buffer = np.zeros((self.size, self.observation_dimensions), dtype=np.float32)
#         self.action_buffer = np.zeros((self.size, self.num_components), dtype=np.float32)
#         self.advantage_buffer = np.zeros(self.size, dtype=np.float32)
#         self.reward_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
#         self.return_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
#         self.value_buffer = np.zeros((self.size, self.num_objectives), dtype=np.float32)
#         self.logprobability_buffer = np.zeros(self.size, dtype=np.float32)
#         self.counter, self.trajectory_start_index = 0, 0
#
#     def store(self, observation, action, reward, value, logprobability):
#         """
#         Append one step of agent-environment interaction
#         :param observation: ndarray - Believe matrix (num_components x (num_states + 1))
#         :param action: ndarray - Actions for the specific timestep (num_components)
#         :param reward: float - Reward for the specific timestep
#         :param value: float - Value for the specific timestep
#         :param logprobability: ndarray - Log probabilities for selected actions (num_components x num_states)
#         """
#         self.observation_buffer[self.counter] = observation
#         self.action_buffer[self.counter] = action
#         self.reward_buffer[self.counter] = reward
#         self.value_buffer[self.counter] = value
#         self.logprobability_buffer[self.counter] = logprobability.detach()
#         self.counter += 1
#
#     def finish_trajectory(self, last_value=0, w_rewards=None):
#         # Finish the trajectory by computing advantage estimates and rewards-to-go
#         path_slice = slice(self.trajectory_start_index, self.counter)
#         rewards = np.append(self.reward_buffer[path_slice], [[last_value] * self.num_objectives], axis=0)
#         # rewards_dot = np.einsum('ij,j->i', rewards, w_rewards)
#
#         values = np.append(self.value_buffer[path_slice], [[last_value] * self.num_objectives], axis=0)
#         values_dot = np.einsum('ij,j->i', values, w_rewards)
#         # deltas = (rewards_dot + self.gamma * np.roll(values_dot, -1) - values_dot)[:-1]
#
#         # self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma * self.lam)
#         self.return_buffer[path_slice] = self.discounted_cumulative_sums(rewards, self.gamma)[:-1]
#         returns_dot = np.einsum('ij,j->i', self.return_buffer[path_slice], w_rewards)
#
#         # Simplification
#         self.advantage_buffer[path_slice] = returns_dot - values_dot[:-1]
#         self.trajectory_start_index = self.counter
#
#     def get(self):
#         # Get all data of the buffer
#         return (
#             self.observation_buffer,
#             self.action_buffer,
#             self.advantage_buffer,
#             self.return_buffer,
#             self.logprobability_buffer,
#         )
#
#     @staticmethod
#     def discounted_cumulative_sums(x, discount):
#         # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
#         return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
#         # This is equivalent to:
#         # summ = [x[-1]]
#         # for val in reversed(x[:-1]):
#         #     summ.append(summ[-1] * discount + val)
#         # return summ[::-1]


class Runner(mp.Process):
    """ Implements a simple single-thread runner class. """

    def __init__(self, name, buffer, pipe, num_episodes, actor, critic, w_rewards, env_name, log_interval):

        mp.Process.__init__(self, name=name)
        print(f"Initializing process {name}")

        # Initialize the environment for the runner
        self.env = gym.make(env_name, quiet=True)
        self.env.reset()
        # The updated rewards for this execution
        self.env.rewards = w_rewards
        self.w_rewards = w_rewards

        # Process-specific initializations
        self.proc_id = name
        self.pipe = pipe

        # Shared objects
        self.buffer = buffer
        self.actor = actor
        self.critic = critic

        # Initialize buffer for this runner
        self.my_transition_buffer = deepcopy(self.buffer) #ind_buffer
        self.my_transition_buffer.reset_buffer()

        self.episode_count = 0
        self.num_episodes = num_episodes
        self.log_interval = log_interval

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()

    def run(self):
        msg_recv = MsgTrainMode(True) # Start with training
        while self.episode_count < self.num_episodes:

            # Initialize environment and buffer
            self.env.w_rewards = self.w_rewards
            self.my_transition_buffer.reset_buffer()

            # Iterate over time steps
            cur_timestep = 0
            # total_urgent_comps = 0
            total_rewards = 0
            while cur_timestep < self.env.timesteps:
                observation = self.env.states_nn
                # Sample action from actor, and value from critic
                with th.no_grad():
                    action, log_prob = self.actor.sample_action_actor(self.env.states_nn, self.episode_count)
                    value = self.critic.sample_action_critic(self.env.states_nn, self.episode_count, self.env.w_rewards)

                # Perform a step into the environment
                observation_new, reward, done, metadata = self.env.step(self.env.actions[copy(action)])
                total_rewards += reward
                # total_urgent_comps += len(self.env.urgent_comps)

                # Store the observation, action, reward, predicted value and log probabilities
                self.my_transition_buffer.store(observation, action, reward, value, log_prob)

                # Check if the episode has ended. If so, proceed to training
                if done or (cur_timestep == self.env.timesteps - 1):
                    observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float)
                    last_value = 0 if done else self.critic(observation_tensor.reshape(1, -1)).item()
                    self.my_transition_buffer.finish_trajectory(last_value, self.env.w_rewards)
                    self.env.reset()

                    act, counts = np.unique(self.my_transition_buffer.action_buffer[:self.env.timesteps], return_counts=True)
                    # if not quiet_glob:
                    print(
                    f"Returns: {self.my_transition_buffer.return_buffer[0].numpy() * self.env.norm_factor} "
                    f"Actions percentages: {dict(zip(act.astype(int), counts * 100 // (self.env.num_components * self.my_transition_buffer.counter)))}"
                    )

                    # Send message to plot rewards:
                    if self.episode_count % self.log_interval == 0 and int(self.proc_id) == 0 and msg_recv.train:
                        msg = MsgRewardInfo(self.proc_id, self.episode_count, metadata)
                        self.pipe.send(msg)
                    break

                cur_timestep += 1

            # Update shared buffer with this runner trajectory
            start_idx = int(self.proc_id) * self.env.timesteps
            end_idx = start_idx + self.env.timesteps

            self.buffer.store_all(self.my_transition_buffer.observation_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.action_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.advantage_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.reward_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.return_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.value_buffer[:self.env.timesteps],
                                  self.my_transition_buffer.logprobability_buffer[:self.env.timesteps],
                                  start_idx=start_idx,
                                  end_idx=end_idx)

            # Send message that episode is completed
            msg = MsgUpdateRequest(int(self.proc_id), self.episode_count, self.buffer, True)
            self.pipe.send(msg)

            msg_recv = self.pipe.recv()  # Wait here until we receive a message to continue

            # add one more episode to count
            if msg_recv.train:
                self.episode_count += 1

        # Send message that runners is completed
        msg = MsgMaxReached(self.proc_id, True)
        self.pipe.send(msg)


class MindmapPPOMultithread(MindmapPPO):
    """
    The main class containing the MindmapPPO algorithm
    """

    def __init__(self, param_file="src/model_params_mt.yaml", quiet=False):

        super().__init__(param_file, quiet)
        global quiet_glob
        quiet_glob = self.quiet

        if not self.multirunner:
            self.processes = 1

        # Initialize Rollout buffer
        self.buffer = MindmapRolloutBufferMultithread(self.env.num_states_iri, self.env.num_components, self.env.timesteps,
                                                      self.env.num_objectives, self.gamma, self.lam,
                                                      self.processes, self.device)

    def run(self, exec_mode, checkpoint=None, reuse_mode="full", max_val=None):

        # Configure the execution mode
        if exec_mode == "train":
            self.run_training(max_val=max_val)
        elif exec_mode == "test":
            self._load_model_weights(checkpoint_dir=checkpoint[0], checkpoint_ep=checkpoint[1], reuse_mode=reuse_mode)
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critic_old.load_state_dict(self.critic.state_dict())
            self.run_testing()

        elif exec_mode == "continue_training":
            self._load_model_weights(checkpoint_dir=checkpoint[0], checkpoint_ep=checkpoint[1], reuse_mode=reuse_mode)
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critic_old.load_state_dict(self.critic.state_dict())
            self.run_training(max_val=max_val)
        else:
            raise ValueError("Choose an execution mode between train, continue_train and test.")

    def run_training(self, max_val=None):

        # starting agents and pipes
        agents = []
        pipes = []

        # tracking subprocess request status
        update_request = [False] * self.processes
        agent_completed = [False] * self.processes

        # tracking training status
        update_iteration = 0

        # Initialize subproceses experience
        for agent_id in range(self.processes):
            p_start, p_end = mp.Pipe()
            agent = Runner(str(agent_id), self.buffer, p_end, self.n_epochs, self.actor_old, self.critic_old,
                           self.env.w_rewards, self.env_name, self.log_interval)
            agent.start()
            agents.append(agent)
            pipes.append(p_start)

        stopped_testing = False # Flag to trigger transition from test to training mode
        msg_send = MsgTrainMode(True)  # Start with training
        # Start training loop
        while True:
            for i, conn in enumerate(pipes):
                if conn.poll():
                    msg = conn.recv()

                    # Ιf agent reached maximum training episode limit
                    if type(msg).__name__ == "MsgMaxReached":
                        agent_completed[i] = True

                    # Ιf agent is waiting for network update
                    elif type(msg).__name__ == "MsgUpdateRequest":
                        update_request[i] = True
                        if False not in update_request:
                            # Check if we are in training mode
                            if msg_send.train:
                                print(f"Episode {update_iteration}")
                                self.buffer = msg.buffer
                                actor_loss, critic_loss = self.train()
                                self.log_after_training(msg.episode, actor_loss, critic_loss)

                                if msg.episode % self.checkpoint_interval == 0 or msg.episode == self.n_epochs - 1:
                                    self.actor.save_checkpoint(msg.episode)
                                    self.critic.save_checkpoint(msg.episode)

                                # Start testing mode
                                if (update_iteration % self.log_interval == 0 or msg.episode == self.n_epochs - 1) \
                                        and not stopped_testing:
                                    if not self.quiet: print(f"Beginning test runs with current weights.")

                                    msg_send = MsgTrainMode(False)
                                    test_returns = []
                                else:
                                    update_iteration += 1
                                    stopped_testing = False

                            # Check if we are in testing mode
                            elif not msg_send.train:
                                test_return = msg.buffer.return_buffer[np.arange(0, self.processes*self.env.timesteps,
                                                                                 self.env.timesteps)]
                                for ret in test_return:
                                    test_returns.append(np.dot(ret.numpy(), self.env.w_rewards))

                                if len(test_returns) > 50:
                                    avg_returns = np.mean(test_returns)
                                    self.total_rewards_test.append(avg_returns)
                                    self.log_after_test_episode(avg_returns, msg.episode)

                                    # Stop testing mode and continue training
                                    if not self.quiet: print(f"Average returns: {np.mean(test_returns)}. "
                                                             f"Continuing training")

                                    if avg_returns < max_val:
                                        print(f"Optimal return {avg_returns} found for PPO based on CBM benchmark "
                                              f"in episode {msg.episode}. Stopping execution. ")
                                    stopped_testing = True

                            # Reset update monitor and send to signal subprocesses to continue
                            update_request = [False] * self.processes
                            for pipe in pipes:
                                pipe.send(msg_send)

                            if stopped_testing:
                                msg_send = MsgTrainMode(True)

                    # if agent is sending over reward stats
                    elif type(msg).__name__ == "MsgRewardInfo": # Log train episode to Tensorboard
                        returns = np.sum(self.buffer.return_buffer[0].numpy() * self.env.w_rewards)
                        values = np.dot(self.critic(th.Tensor(self.env.states_nn)).detach().numpy(), self.env.w_rewards)
                        self.log_after_train_episode(msg.episode, returns, values, {"metadata": msg.metadata})

            if False not in agent_completed:
                print("=Training ended with Max Episodes=")
                break

        for agent in agents:
            agent.terminate()

        # Clear the checkpoints that do not correspond to the best weights
        self.clear_bad_checkpoints()

    def run_testing(self):
        msg_send = MsgTrainMode(False)
        test_returns = {}
        test_actions = {}
        test_rewards = {}

        p_start, p_end = mp.Pipe()
        agent = Runner(str(0), self.buffer, p_end, 2, self.actor_old, self.critic_old,
                       self.env.w_rewards, self.env_name)
        agent.start()

        counter = 0
        test_episodes = 10
        while counter < test_episodes:
            if p_start.poll():
                msg = p_start.recv()
                if type(msg).__name__ == "MsgUpdateRequest":
                    test_return = msg.buffer.return_buffer[0]
                    test_action = msg.buffer.action_buffer
                    test_reward = msg.buffer.reward_buffer

                    label = f"Test {counter}"
                    test_returns[label] = np.dot(test_return.numpy(), self.env.w_rewards)
                    test_actions[label] = test_action[:self.env.timesteps]
                    test_rewards[label] = th.cumsum(test_reward[:self.env.timesteps], dim=0)
                    counter += 1

                    p_start.send(msg_send)

        avg_returns = np.mean(list(test_returns.values()))
        print(f"Average test returns: {avg_returns}")
        self.log_after_testing(test_returns, test_actions, test_rewards)

        # Terminate the agent
        agent.terminate()

    def run_episodes(self, exec_mode="train", checkpoint_dir=None, checkpoint_ep=None, reuse_mode="full"):
        # if w_rewards:
        #     self.env.w_rewards = w_rewards

        # Iterate over episodes
        # If we are in training mode
        if (exec_mode == "train") or (exec_mode == "continue_training"):

            if exec_mode == "continue_training":
                self._load_model_weights(checkpoint_dir, checkpoint_ep, reuse_mode)

            if not self.quiet: print(f"Starting training.")
            for episode in range(self.n_epochs):
                # if not self.quiet:
                print(f"Episode {episode}:")
                self.buffer.reset_buffer()
                transition_buffers_list, metadata_dicts_list = self.runner.run(self.env.timesteps, blueprint=self.buffer,
                                                          transition_buffer_dict={"transition_buffer": self.buffer})

                if isinstance(transition_buffers_list, dict):
                    transition_buffers_list = [transition_buffers_list]

                buff_count = 0
                for buffer in transition_buffers_list:
                    buff = buffer["transition_buffer"]

                    self.buffer.observation_buffer[
                    buff_count:buff_count + self.env.timesteps] = buff.observation_buffer[:self.env.timesteps]
                    self.buffer.action_buffer[buff_count:buff_count + self.env.timesteps] = buff.action_buffer[
                                                                                            :self.env.timesteps]
                    self.buffer.advantage_buffer[buff_count:buff_count + self.env.timesteps] = buff.advantage_buffer[
                                                                                               :self.env.timesteps]
                    self.buffer.reward_buffer[buff_count:buff_count + self.env.timesteps] = buff.reward_buffer[
                                                                                            :self.env.timesteps]
                    self.buffer.return_buffer[buff_count:buff_count + self.env.timesteps] = buff.return_buffer[
                                                                                            :self.env.timesteps]
                    self.buffer.value_buffer[buff_count:buff_count + self.env.timesteps] = buff.value_buffer[
                                                                                           :self.env.timesteps]
                    self.buffer.logprobability_buffer[
                    buff_count:buff_count + self.env.timesteps] = buff.logprobability_buffer[:self.env.timesteps]
                    buff_count += self.env.timesteps

                #                 _ = self.run_episode(episode, train_phase="learn")
                # log everything that is needed
                if episode % self.checkpoint_interval == 0 or episode == self.n_epochs - 1:
                    self.actor.save_checkpoint(episode)
                    self.critic.save_checkpoint(episode)

                # returns = np.sum(np.einsum('ij,j->i', buff.reward_buffer, self.env.w_rewards * self.env.norm_factor))
                returns = np.sum(buff.return_buffer[0] * self.env.w_rewards)
                values = np.dot(self.critic(th.Tensor(self.env.states_nn)).detach().numpy(), self.env.w_rewards)
                # values = self.critic(th.Tensor(self.env.states_nn))
                self.log_after_train_episode(episode, returns, values, metadata_dicts_list[-1])

                # Train the two networks based on the experience of this episode
                actor_loss, critic_loss = self.train()

                self.log_after_training(episode, actor_loss, critic_loss)

                if self.test_interval:
                    if episode % self.test_interval == 0 or episode == self.n_epochs - 1:
                        if not self.quiet: print(f"Beginning test runs with current weights.")
                        test_rewards = []
                        for test_episode in range(self.test_n_epochs):
                            test_rewards.append(self.run_episode(test_episode, train_phase="test_train"))
                        if exec_mode != "test":
                            self.total_rewards_test.append(np.mean(test_rewards))
                            self.log_after_test_episode(np.mean(test_rewards), episode)

                # CRITIC ERROR
                # if np.sqrt(np.abs(critic_loss.detach().numpy()/returns)) < 0.01:
                #     print(f"STOPPING EXECUTION DUE TO CONVERGENCE OF RETURNS AND VALUES IN EPISODE {episode}.")
                #     print(f"Critic loss is {critic_loss.detach().numpy()} and returns are {returns}.")
                #     break

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


if __name__ == "__main__":
    os.chdir("../")

    start = time.time()
    ppo = MindmapPPOMultithread()
    print(f"Weights are {ppo.env.w_rewards}")
    ppo.run(exec_mode="train")
    # ppo.run(exec_mode="test", checkpoint=("src/model_weights/20230316163006_289/", 10500))
    # ppo.run(exec_mode="continue_training", checkpoint=("src/model_weights/20230316163006_289/", 10500))
    print(f"Total time {time.time() - start} seconds.")
