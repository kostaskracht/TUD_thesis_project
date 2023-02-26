import os
from copy import copy, deepcopy
import time
import torch as th
import numpy as np
import gym
import threading
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import thesis_env
from PPO import MindmapRolloutBuffer, MindmapPPO


class MindmapRolloutBufferMultithread(MindmapRolloutBuffer):
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

    def __init__(self, num_states, num_actions, num_components, timesteps, num_objectives, gamma=0.95, lam=0.95,
                 processes=1):
        super().__init__(num_states, num_actions, num_components, timesteps,
                         num_objectives, gamma, lam)

        self.processes = processes
        self.size = timesteps * self.processes
        # self.lock = threading.Lock()


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

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()

    def run_test(self, num, transition_buffer_dicts):
        transition_buffer_dicts.update({str(np.random.choice(np.arange(10000))): num})
        return transition_buffer_dicts

    def run(self, n_steps, blueprint, transition_buffer_dict=None, i=None):
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

                act, counts = np.unique(my_transition_buffer.action_buffer[:self.env.timesteps], return_counts=True)
                print(
                    #                     f"{train_phase} episode: {episode}, Total return:"
                    f"{np.sum(my_transition_buffer.reward_buffer, axis=0) * self.env.norm_factor} "
                    # f" {my_transition_buffer.reward_buffer.sum() * self.env.norm_factor[0]} "
                    f"Actions percentages {dict(zip(act.astype(int), counts * 100 // (self.env.num_components * my_transition_buffer.counter)))}"
                    # f"Total urgent comps {total_urgent_comps}"
                )

                # if not transition_buffer_dict:
                #     transition_buffer = my_transition_buffer
                break

            cur_timestep += 1

        # transition_buffer_dict.update({"transition_buffer": my_transition_buffer})
        if i:
            transition_buffer_dict[f"transition_buffer_{i}"] = my_transition_buffer
        else:
            transition_buffer_dict[f"transition_buffer"] = my_transition_buffer
        # return transition_buffer_dict

    def run_episode(self, transition_buffer=None, trim=True, return_dict=None):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statistics. """
        return self.run(0, transition_buffer)

def run_test(num, transition_buffer_dicts):
    transition_buffer_dicts[str(np.random.choice(np.arange(10000)))] = num
    print(transition_buffer_dicts)
    # return transition_buffer_dicts

class MultiRunner:
    """ Simple class that runs multiple Runner objects in parallel and merges their outputs. """

    def __init__(self, controller, processes, fork_on="thread"):
        self.workers = []
        self.runners = []
        n = processes
        for _ in range(n):
            self.runners.append(Runner(controller=copy(controller)))

        self.controller = controller
        self.fork_on = fork_on

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        # Join all workers
        for w in self.workers:
            w.join()
        # Exit all environments
        for r in self.runners:
            r.close()

    def fork_thread(self, target, common_args=None, specific_args=None):
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

    def fork_process(self, target, common_args=None, specific_args=None):
        """ Executes the function "target" on all runners. "common_args" is a dictionary of
            arguments that are passed to all runners, "specific_args" is a list of
            dictionaries that contain individual parameters for each runner. """
        # Fork all runners
        self.workers = []
        for i, r in enumerate(self.runners):
            r_args = [] if specific_args is None else [arg[i] for arg in specific_args]
            # r_args = [] if specific_args is None else [specific_args[i]]
            # process = multiprocessing.Process(target=target, args=(r, *common_args, *r_args))
            process = multiprocessing.Process(target=r.run, args=(*common_args, *r_args, i))
            self.workers.append(process)
            process.start()

        # Join all runners
        for w in self.workers:
            w.join()


    def run(self, n_steps, blueprint, transition_buffer_dict):
        """ Runs n_steps, split amongst runners, and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        n_steps = n_steps // len(self.runners)

        if self.fork_on == "thread":
            transition_buffer_dicts = [{} for _ in self.runners]
            self.fork_thread(target=Runner.run, common_args=(n_steps, blueprint), specific_args=(transition_buffer_dicts,))
            # self.fork(target=Runner.run, common_args=(n_steps, transition_buffer))
        elif self.fork_on == "process":
            manager = multiprocessing.Manager()
            transition_buffer_dicts = manager.dict()
            # self.fork_process(target=Runner.run, common_args=(n_steps, blueprint),
            #                   specific_args=(transition_buffer_dicts,))
            # self.fork_process(target=Runner.run_test, common_args=(5, transition_buffer_dicts,))
            self.fork_process(target=Runner.run, common_args=(n_steps, blueprint, transition_buffer_dicts,))
            transition_buffer_dicts = [{"transition_buffer": val} for val in transition_buffer_dicts.values()]
        else:
            raise TypeError("Wrong fork_on setting. Available options are 'thread' and 'process'")

        return transition_buffer_dicts


class MindmapPPOMultithread(MindmapPPO):
    """
    The main class containing the MindmapPPO algorithm
    """

    def __init__(self, param_file="src/model_params_mt.yaml", quiet=False):

        super().__init__(param_file, quiet)

        if not self.multirunner:
            self.processes = 1

        # Initialize Rollout buffer
        self.buffer = MindmapRolloutBufferMultithread(self.env.num_states_iri, self.env.num_actions,
                                                       self.env.num_components, self.env.timesteps,
                                                       self.env.num_objectives,
                                                       self.gamma, self.lam, self.processes)

        # Multi-threading execution
        self.runner = MultiRunner(self, processes=self.processes, fork_on=self.fork_on) if (self.multirunner == True) \
            else Runner(self)

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
                transition_buffers_list = self.runner.run(self.env.timesteps, blueprint=self.buffer,
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

                returns = np.sum(np.einsum('ij,j->i', buff.reward_buffer, self.env.w_rewards * self.env.norm_factor))
                self.log_after_train_episode(episode, returns)

                # Train the two networks based on the experience of this episode
                actor_loss, critic_loss = self.train()

                self.log_after_training(episode, actor_loss, critic_loss)

                if self.test_interval:
                    if episode % self.test_interval == 0 or episode == self.n_epochs - 1:
                        print(f"Beginning test runs with current weights.")
                        test_rewards = []
                        for test_episode in range(self.test_n_epochs):
                            test_rewards.append(self.run_episode(test_episode, train_phase="test_train"))
                        if exec_mode != "test":
                            self.total_rewards_test.append(np.mean(test_rewards))
                            self.log_after_test_episode(np.mean(test_rewards), episode)

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


if __name__ == "__main__":
    os.chdir("../")

    start = time.time()
    ppo = MindmapPPOMultithread()
    print(f"Weights are {ppo.env.w_rewards}")
    ppo.run_episodes(exec_mode="train")
    ppo.runner.close()
    print(f"Total time {time.time() - start} seconds.")
