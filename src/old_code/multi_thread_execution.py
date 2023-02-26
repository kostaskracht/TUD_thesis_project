import gym
import numpy as np
import torch as th
import numbers
import threading

from PPO import MindmapRolloutBuffer
from PPO import MindmapPPO


class Runner:
    """ Implements a simple single-thread runner class. """

    def __init__(self, controller, params={}, exploration_step=1):
        self.env = gym.make(params.get('env', 'thesis-env-v1'))
        self.cont_actions = isinstance(self.env.action_space, gym.spaces.Box)
        self.controller = controller
        self.epi_len = params.get('max_episode_length', self.env._max_episode_steps)
        self.gamma = params.get('gamma', 0.99)
        self.state_shape = self.env.observation_space.shape

        self.buffer = MindmapRolloutBuffer(5, 5, 12, 20, 0.99, 0.97)

        # Set up current state and time step
        self.sum_rewards = 0
        self.state = None
        self.time = 0
        self._next_step()

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()

    # def transition_format(self):
    #     """ Returns the format of transtions: a dictionary of (shape, dtype) entries for each key. """
    #     return {'actions': ((1,), th.long),
    #             'states': (self.state_shape, th.float32),
    #             'next_states': (self.state_shape, th.float32),
    #             'rewards': ((1,), th.float32),
    #             'dones': ((1,), th.bool),
    #             'returns': ((1,), th.float32)}

    # def _wrap_transition(self, s, a, r, ns, d):
    #     """ Takes a transition and returns a corresponding dictionary. """
    #     trans = {}
    #     form = self.transition_format()
    #     for key, val in [('states', s), ('actions', a), ('rewards', r), ('next_states', ns), ('dones', d)]:
    #         if not isinstance(val, th.Tensor):
    #             if isinstance(val, numbers.Number) or isinstance(val, bool): val = [val]
    #             val = th.tensor(val, dtype=form[key][1])
    #         if len(val.shape) < len(form[key][0]) + 1: val = val.unsqueeze(dim=0)
    #         trans[key] = val
    #     return trans

    def _run_step(self, a):
        """ Make a step in the environment (and update internal bookeeping) """
        ns, r, d, _ = self.env.step(a.item())
        self.sum_rewards += r
        # if self.use_pixels: ns = self._pixel_observation()
        return r, ns, d

    def _next_step(self, done=True, next_state=None):
        """ Switch to the next time-step (and update internal bookeeping) """
        self.time = 0 if done else self.time + 1
        if done:
            self.sum_rewards = 0
            self.state = self.env.reset()
            # if self.use_pixels: self.state = self._pixel_observation(reset=True)
        else:
            self.state = next_state

    def run(self, episode, train_phase="learn"):
        # Initialize buffer
        self.buffer.reset_buffer()

        # Iterate over time steps
        cur_timestep = 0
        # total_urgent_comps = 0
        while cur_timestep < self.env.timesteps:
            observation = self.env.states_nn
            # Sample action from actor, and value from critic
            action, log_prob, value = self.sample_action(self.env.states_nn)

            # Perform a step into the environment
            from copy import copy
            observation_new, reward, done, _ = self.env.step(self.env.actions[copy(action)])

            # TODO scalarize the reward function
            reward = reward[0]
            # total_urgent_comps += len(self.env.urgent_comps)

            # Store the observation, action, reward, predicted value and log probabilities
            self.buffer.store(observation, action, reward, value, log_prob)
            observation_new = observation

            # Check if the episode has ended. If so, proceed to training
            if done or (cur_timestep == self.env.timesteps - 1):
                observation_tensor = th.tensor(np.array([observation_new]), dtype=th.float).to(
                    self.device)
                last_value = 0 if done else self.critic(observation_tensor.reshape(1, -1)).item()
                self.buffer.finish_trajectory(last_value, self.env.w_rewards)
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

                act, counts = np.unique(self.buffer.action_buffer, return_counts=True)
                print(f"{train_phase} episode: {episode}, Total return:"
                      f" {self.buffer.reward_buffer.sum() * self.env.norm_factor[0]} "
                      f"Actions percentages {dict(zip(act.astype(int), counts*100//(self.env.num_components*self.env.timesteps)))}"
                      # f"Total urgent comps {total_urgent_comps}"
                      )
                break

            cur_timestep += 1

        return self.buffer

    # def run(self, n_steps, transition_buffer=None, trim=True, return_dict=None):
    #     """ Runs n_steps in the environment and stores them in the trainsition_buffer (newly created if None).
    #         If n_steps <= 0, stops at the end of an episode and optionally trins the transition_buffer.
    #         Returns a dictionary containing the transition_buffer and episode statstics. """
    #     # my_transition_buffer = TransitionBatch(n_steps if n_steps > 0 else self.epi_len, self.transition_format())
    #     my_transition_buffer = MindmapRolloutBuffer(5, 5, 12, 20, 0.99, 0.97)
    #     time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
    #     max_steps = n_steps if n_steps > 0 else self.epi_len
    #     for t in range(max_steps):
    #         # One step in the envionment
    #         a = self.controller.choose(self.state)
    #         r, ns, d = self._run_step(a)
    #         terminal = d and self.time < self.epi_len - 1
    #         my_transition_buffer.add(self._wrap_transition(self.state, a, r, ns, terminal))
    #         if t == self.epi_len - 1: d = True
    #         # Compute discounted returns if episode has ended or max_steps has been reached
    #         if d or t == (max_steps - 1):
    #             my_transition_buffer['returns'][t] = my_transition_buffer['rewards'][t]
    #             for i in range(t - 1, episode_start - 1, -1):
    #                 my_transition_buffer['returns'][i] = my_transition_buffer['rewards'][i] \
    #                                                      + self.gamma * my_transition_buffer['returns'][i + 1]
    #             episode_start = t + 1
    #         # Remember statistics and advance (potentially initilaizing a new episode)
    #         if d:
    #             episode_lengths.append(self.time + 1)
    #             episode_rewards.append(self.sum_rewards)
    #         self._next_step(done=d, next_state=ns)
    #         time += 1
    #         # If n_steps <= 0, we return after one episode (trimmed if specified)
    #         if d and n_steps <= 0:
    #             my_transition_buffer.trim()
    #             break
    #     # Add the sampled transitions to the given transition buffer
    #     transition_buffer = my_transition_buffer if transition_buffer is None \
    #         else transition_buffer.add(my_transition_buffer)
    #     if trim: transition_buffer.trim()
    #     # Return statistics (mean reward, mean length and environment steps)
    #     if return_dict is None: return_dict = {}
    #     return_dict.update({'buffer': transition_buffer,
    #                         'episode_reward': None if len(episode_rewards) == 0 else np.mean(episode_rewards),
    #                         'episode_length': None if len(episode_lengths) == 0 else np.mean(episode_lengths),
    #                         'env_steps': time})
    #     return return_dict

    # def run_episode(self, transition_buffer=None, trim=True, return_dict=None):
    def run_episode(self, episode=None, train_phase="learn"):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        # return self.run(0, transition_buffer, trim, return_dict)
        return self.run(episode, train_phase)


class MultiRunner:
    """ Simple class that runs multiple Runner objects in parallel and merges their outputs. """

    def __init__(self, controller, params={}):
        self.workers = []
        self.runners = []
        n = params.get('parallel_environments', 1)
        for _ in range(n):
            self.runners.append(Runner(controller=controller, params=params))

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

    def run(self, n_steps, transition_buffer=None, trim=True):
        """ Runs n_steps, split amongst runners, and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        n_steps = n_steps // len(self.runners)
        if transition_buffer is None:
            buffer_len = len(self.runners) * (n_steps if n_steps > 0 else self.runners[0].epi_len)
            transition_buffer = TransitionBatch(buffer_len, self.runners[0].transition_format())
        return_dicts = [{} for _ in self.runners]
        self.fork(target=Runner.run, common_args=(n_steps, transition_buffer, False), specific_args=(return_dicts,))
        if trim: transition_buffer.trim()
        rewards = [d['episode_reward'] for d in return_dicts if d['episode_reward'] is not None]
        lengths = [d['episode_length'] for d in return_dicts if d['episode_reward'] is not None]
        return {'buffer': transition_buffer,
                'episode_reward': np.mean(rewards) if len(rewards) > 0 else None,
                'episode_length': np.mean(lengths) if len(lengths) > 0 else None,
                'env_steps': len(transition_buffer)}

    def run_episode(self, transition_buffer=None, trim=True):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        return self.run(0, transition_buffer, trim)

class ACController:
    """ Controller for Q-value functions, synchronizes the model calls. """

    def __init__(self, model, num_actions=None, params={}):
        self.lock = threading.Lock()
        self.num_actions = model[-1].out_features if num_actions is None else num_actions
        self.model = model

    def copy(self):
        """ Shallow copy of this controller that does not copy the model. """
        return ACController(model=self.model, num_actions=self.num_actions)

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
        try: mx = observation if precomputed else self.model(self.sanitize_inputs(observation))[:, :self.num_actions]
        finally: self.lock.release()
        return th.nn.functional.softmax(mx, dim=-1)

    def choose(self, observation, **kwargs):
        return th.distributions.Categorical(probs=self.probabilities(observation, **kwargs)).sample()


class Learner:
    """ A learner that performs a version of PPO. """

    def __init__(self, model, controller=None, params={}):
        self.model = model
        self.controller = controller
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 0)
        self.all_parameters = list(model.parameters())
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.compute_next_val = False  # whether the next state's value is computed
        self.old_pi = None  # this variable can be used for your PPO implementation

    def set_controller(self, controller):
        """ This function is called in the experiment to set the controller. """
        self.controller = controller

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        return batch['returns']

    def _value_loss(self, batch, values=None, next_values=None):
        """ Computes the value loss (if there is one). """
        return 0

    def _policy_loss(self, pi, advantages):
        """ Computes the policy loss. """
        return -(advantages.detach() * pi.log()).mean()

    # def train(self, batch):
    #     assert self.controller is not None, "Before train() is called, a controller must be specified. "
    #     self.model.train(True)
    #     self.old_pi, loss_sum = None, 0.0
    #     for _ in range(1 + self.offpolicy_iterations):
    #         # Compute the model-output for given batch
    #         out = self.model(batch['states'])  # compute both policy and values
    #         val = out[:, -1].unsqueeze(dim=-1)  # last entry are the values
    #         next_val = self.model(batch['next_states'])[:, -1].unsqueeze(dim=-1) if self.compute_next_val else None
    #         pi = self.controller.probabilities(out[:, :-1], precomputed=True).gather(dim=-1, index=batch['actions'])
    #         # Combine policy and value loss
    #         loss = self._policy_loss(pi, self._advantages(batch, val, next_val)) \
    #                + self.value_loss_param * self._value_loss(batch, val, next_val)
    #         # Backpropagate loss
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
    #         self.optimizer.step()
    #         loss_sum += loss.item()
    #     return loss_sum

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

        return actor_loss


class Experiment:
    """ Abstract class of an experiment. Contains logging and plotting functionality."""

    def __init__(self, params, model, learner=None, **kwargs):
        self.params = params
        self.plot_frequency = params.get('plot_frequency', 100)
        self.plot_train_samples = params.get('plot_train_samples', True)
        self.print_when_plot = params.get('print_when_plot', False)
        self.print_dots = params.get('print_dots', False)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_losses = []
        self.env_steps = []
        self.total_run_time = 0.0
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ACController(model, num_actions=gym.make(params['env']).action_space.n, params=params)
        self.runner = MultiRunner(self.controller, params=params) if params.get('multi_runner', True) \
            else Runner(self.controller, params=params)
        self.learner = Learner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)

    # def plot_training(self, update=False):
    #     """ Plots logged training results. Use "update=True" if the plot is continuously updated
    #         or use "update=False" if this is the final call (otherwise there will be double plotting). """
    #     # Smooth curves
    #     window = max(int(len(self.episode_returns) / 50), 10)
    #     if len(self.episode_losses) < window + 2: return
    #     returns = np.convolve(self.episode_returns, np.ones(window) / window, 'valid')
    #     lengths = np.convolve(self.episode_lengths, np.ones(window) / window, 'valid')
    #     losses = np.convolve(self.episode_losses, np.ones(window) / window, 'valid')
    #     env_steps = np.convolve(self.env_steps, np.ones(window) / window, 'valid')
    #     # Determine x-axis based on samples or episodes
    #     if self.plot_train_samples:
    #         x_returns = env_steps
    #         x_losses = env_steps[(len(env_steps) - len(losses)):]
    #     else:
    #         x_returns = [i + window for i in range(len(returns))]
    #         x_losses = [i + len(returns) - len(losses) + window for i in range(len(losses))]
    #     # Create plot
    #     colors = ['b', 'g', 'r']
    #     fig = plt.gcf()
    #     fig.set_size_inches(16, 4)
    #     plt.clf()
    #     # Plot the losses in the left subplot
    #     pl.subplot(1, 3, 1)
    #     pl.plot(env_steps, returns, colors[0])
    #     pl.xlabel('environment steps' if self.plot_train_samples else 'episodes')
    #     pl.ylabel('episode return')
    #     # Plot the episode lengths in the middle subplot
    #     ax = pl.subplot(1, 3, 2)
    #     ax.plot(env_steps, lengths, colors[0])
    #     ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
    #     ax.set_ylabel('episode length')
    #     # Plot the losses in the right subplot
    #     ax = pl.subplot(1, 3, 3)
    #     ax.plot(x_losses, losses, colors[0])
    #     ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
    #     ax.set_ylabel('loss')
    #     # dynamic plot update
    #     display.clear_output(wait=True)
    #     if update:
    #         display.display(pl.gcf())

    def close(self):
        """ Frees all allocated runtime ressources, but allows to continue the experiment later.
            Calling the run() method after close must be able to pick up the experiment where it was. """
        pass

    def run(self):
        """ Starts (or continues) the experiment. """
        assert False, "You need to extend the Expeirment class and override the method run(). "

