import gym
from gym import spaces
import yaml
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import copy, deepcopy
import logging

from thesis_env.envs.assignment import load_network, assignment_loop, BPRcostFunction
from thesis_env.envs.visualize_road_network import plot_graph


class ThesisEnv(gym.Env):
    """ ThesisEnv
    This is the base class for a road network environment with arbitrary parameters.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, param_file="environments/env_params.yaml",
                 action_results="environments/action_results.yaml", quiet=False):

        # Load parameters
        self.quiet = quiet
        self.param_file = param_file
        self.param_dict = self._load_yaml_file(self.param_file)
        self.action_results = self._load_yaml_file(action_results)
        # Load all parameters denoted on the respective yaml file. Variable names will be the key
        # of each parameter
        self._load_params()
        self._load_files()

        # self._construct_obs_prob_matrix() # Unused in the implementation

        # Set the seed for reproducibility
        if self.seed:
            np.random.seed(self.seed)

        # Filter components and actions based on the "components_to_keep" and "actions_to_remove" parameters
        self._filter_components_actions()

        # Define the actions space
        self.action_space = spaces.MultiDiscrete([(self.num_actions)] * self.num_components)

        # Define the state space
        self.observation_space = spaces.Discrete(self.num_states_iri * self.num_components)

        # Compute the total cost per action
        # self.c_act = self.compute_action_cost() # Unused for now

        # Set the transition matrix to be used in each action
        self.transitions = {
            "plain": {"cci": self.tp_cci_plain, "iri": self.tp_iri_plain},
            "minor": {"cci": self.tp_cci_minor, "iri": self.tp_iri_minor},
            "major": {"cci": self.tp_cci_major, "iri": self.tp_iri_major},
            "replace": {"cci": self.tp_cci_replace, "iri": self.tp_iri_replace}
        }

        # Initialize environmental variables
        self.act_ongoing = np.zeros(self.num_components, dtype=bool)
        self.episode_cost = 0

        ## ROAD NETWORK
        # Initialize road as graph network
        self.net_file = "data/init_data/sioux_falls/SiouxFalls_net.csv"
        self.road_network = load_network(net_file=self.net_file, verbose=False)

        # Initialize buffer to keep traffic assignment solutions
        self.traffic_assignment_solutions = {}
        self.num_traffic_assignments = 0

        # Keep only the used network components
        if self.components_to_keep:
            self._filter_road_network_components()

        # Perform basic network tweaking to achieve more interesting results
        for key, link in self.road_network.linkSet.items():
            self.road_network.linkSet[key].beta = 2
            self.road_network.linkSet[key].max_capacity *= 0.5
            self.road_network.linkSet[key].capacity *= 0.5
            self.road_network.linkSet[key].flow_init = 0
            self.road_network.linkSet[key].max_capacity_init = self.road_network.linkSet[key].max_capacity

        for key, trip in self.road_network.tripSet.items():
            self.road_network.tripSet[key].demand *= 5

        # Reset the environment
        self.reset()
        self.time_count = 0

        # Compute the travel time and traffic flows for the initial network
        # self.TSTT_init, self.flows_init = self._execute_traffic_assignment(np.zeros_like(self.components, dtype=bool))
        self.TSTT_init, self.flows_init, self.comp_time, self.comp_fft = self._traffic_assignment(np.zeros_like(self.components, dtype=int),
                                                                   np.ones_like(self.components, dtype=bool))

        # Compute the carbon footprint of the initial network
        # self.carbon_footprint_init = self._compute_carbon_footprint(np.repeat([self.flows_init], repeats=12, axis=0).T)
        self.carbon_footprint_init, _ = self._compute_carbon_footprint(self.flows_init, self.comp_time, self.comp_fft)

        # self.user_cost_init = self._compute_user_cost(self.flows_init)

        # Initialize the road network flows with the initial ones, for faster convergence
        for key, link in self.road_network.linkSet.items():
            self.road_network.linkSet[key].flow_init = self.road_network.linkSet[key].flow

        # Compute normalization factors if they are not set in the yaml file
        if not self.norm_factor:
            self.norm_factor = np.zeros(self.num_objectives)
            self.norm_factor[0] = np.sum(np.max(np.abs(self.c_mai + self.c_ins)[self.actions], axis=1)) / 5
            avg_carbon = np.sum(np.asarray(self.transport_types) * np.asarray(self.carbon_foot_by_type))
            self.norm_factor[1] = np.sum(
                self.comp_len[self.components] * self.capacity[self.components]) * avg_carbon * 12
            self.norm_factor[2] = np.sum(self.comp_len[self.components] * self.capacity[self.components]) * 10 * 12
        else:
            self.norm_factor = np.asarray(self.norm_factor)

        # SOS - A different normalization is computed for every runner!
        if not self.std_reward or not self.max_reward:
            rew_basket = []
            self.std_reward = np.ones(self.num_objectives)
            self.max_reward = np.zeros(self.num_objectives)
            print("Starting normalization episodes")
            for idx in range(self.norm_episodes):
                print(f"Normalization episode {idx + 1}/{self.norm_episodes}")
                self.reset()
                for i in range(self.timesteps):
                    cur_action = np.random.choice(np.arange(self.num_actions), size=self.num_components, replace=True,)
                    _, step_cost, _, _ = self.step(self.actions[cur_action])
                    rew_basket.append(step_cost)
            self.reset()

            self.std_reward = np.std(rew_basket, axis=0)
            self.max_reward = np.max(rew_basket, axis=0)
            self.min_reward = np.min(rew_basket, axis=0)
            self.mean_reward = np.mean(rew_basket, axis=0)
            print(f"Normalization values: std: {self.std_reward}, max: {self.max_reward}")

    def step(self, action: np.ndarray):
        """
        This method performs a step in the environment, given a set of actions (one for each
        component)
        :param action: list(num_components) - Actions indices.
        :return: states: ndarray(num_components, num_states) - States
        :return: cost: float - Total cost of the episode
        :return: done: boolen - True if the episode has ended (not used in our case)
        :return: metadata: dict - Dictionary containing metadata (empty for now)

        """
        # Initialize cost array of step
        step_cost = np.zeros(self.num_objectives)
        done = False

        # Calculate the immediate costs
        cost_action, cost_insp, is_comp_active, act_init, action, cost_risk = self.get_immediate_cost(action)

        # Initialize the tmp variable for ongoing actions
        act_ongoing_tmp = self.act_ongoing.copy()

        # Iterate over components. Assume that maintenance and inspection takes place at the beginning of the timestep.
        for idx, comp in enumerate(self.components):

            # Get current action
            cur_action = action[idx]

            # Calculate the new state distribution and the cost, based on the action given

            # If we are in an ongoing action, reset the "ongoing_action" flag to False for the next step
            if self.act_ongoing[idx]:
                act_ongoing_tmp[idx] = False
            elif (not self.act_ongoing[idx]) and (self.actions_long[idx, cur_action]) and (
                    is_comp_active[idx]):
                act_ongoing_tmp[idx] = True

            # If the component is active, update the deterioration rate and the state based on the action
            if is_comp_active[idx]:
                # Check if the selected action resets the deterioration rate
                if "reset_deter_rate" in self.action_results[cur_action]:
                    self._reset_deter_rate(idx, cur_action)

                # Check if states shift is a result of the selected action
                if "shift_back_deter_rate" in self.action_results[cur_action]:
                    self._shift_deter_rate(idx, shift=self.action_results[cur_action]["shift_back_deter_rate"])

                # Check the type of transition to perform
                if not "transition" in self.action_results[cur_action]:
                    raise ValueError(f"No transition type has been specified for action {cur_action}")
                else:
                    self._update_states(idx, self.transitions[self.action_results[cur_action]["transition"]])

            # If the component is not active, update the states with "do nothing" transition matrix
            elif (not is_comp_active[idx]) and (not self.act_ongoing[idx]):
                self._update_states(idx, self.transitions["plain"])
                cur_action = 0  # if the component is not active, assume that the action is "do nothing"

            # Perform inference (belief update) if:
            # - There is no ongoing action
            # - We aren't starting an ongoing action
            if (not self.act_ongoing[idx]) and (not act_ongoing_tmp[idx]):
                if is_comp_active[idx]:
                    cur_action_fin = cur_action
                else:
                    cur_action_fin = 0  # If component is not active, then perform update with do nothing

                self.states_iri[idx] = self._perform_inference(cur_action_fin, self.states_iri[idx],
                                                               self.obs_probs_iri[idx])

                # obs_dist_iri = self.states_iri[idx] @ self.obs_probs_iri[idx, cur_action_fin]
                # obs_dist_iri = obs_dist_iri/np.sum(obs_dist_iri)
                # obs_iri = np.random.choice(range(self.num_states_iri), size=None, replace=True, p=obs_dist_iri)
                #
                # update_iri = self.states_iri[idx] * self.obs_probs_iri[idx, cur_action_fin, :, obs_iri]
                # self.states_iri[idx] = update_iri / np.sum(update_iri)

                if self.use_cci_state:
                    self.states_cci[idx] = self._perform_inference(cur_action_fin, self.states_cci[idx],
                                                                   self.obs_probs_cci[idx])
                    # obs_dist_cci = self.states_cci[idx] @ self.obs_probs_cci[idx, cur_action_fin]
                    # obs_dist_cci = obs_dist_cci/np.sum(obs_dist_cci)
                    # obs_cci = np.random.choice(range(self.num_states_cci), size=None, replace=True,
                    #                            p=obs_dist_cci)
                    #
                    # update_cci = self.states_cci[idx] * self.obs_probs_cci[idx, cur_action_fin, :,
                    #                                     obs_cci]
                    # self.states_cci[idx] = update_cci / np.sum(update_cci)

        # For the whole system
        if self.is_objective_active[0]:
            # Add up the costs from actions
            step_cost[0] = cost_action + cost_insp + cost_risk

        if self.is_objective_active[1] or self.is_objective_active[2]:
            # Calculate the traffic flows in the network per month
            # comp traffic can be either (components x months)
            total_travel_time, comp_traffic, comp_time, comp_fft = self._traffic_assignment(action, is_comp_active)

            # Visualize the road network
            if self.plot_road_network:
                node_ids_in_use = []
                for idx, coord in enumerate(self.node_coords):
                    if coord[0] in self.edges[self.components]:
                        node_ids_in_use.append(idx)

                metric_to_plot = comp_traffic[:, 0] / self.capacity[self.components]
                title_to_plot = f"Timestep {self.time_count}, Month {'January'}, closed segments {np.where(act_init != 0)[0]}"
                title_to_plot = " "
                min_max_to_plot = (0, np.max(metric_to_plot))

                # Other plot ideas:

                # metric_to_plot = np.zeros(len(self.components))
                # title_to_plot = f"IRI States - Timestep {self.time_count}, Month {'January'}"
                # min_max_to_plot = (0, 5)
                #
                # metric_to_plot = action
                # title_to_plot = f"Actions - Timestep {self.time_count}"
                # min_max_to_plot = (0, 9)

                plot_graph(np.asarray(self.node_coords)[node_ids_in_use], self.edges[self.components],
                           metric_to_plot, title_to_plot, min_max=min_max_to_plot)

        if self.is_objective_active[1]:
            # Calculate the total carbon footprint for this step
            carbon_emissions, emissions_from_condition_perc = self._compute_carbon_footprint(
                comp_traffic, comp_time, comp_fft)  # emissions are negative
            carbon_emissions_from_actions = self._compute_carbon_footprint_actions(action)

            # Log everything
            # Cost components are logged under cost calculation method
            self.carbon_components["rerouting"] += self.gamma ** self.time_count * (
                        carbon_emissions / (1 + emissions_from_condition_perc) - self.carbon_footprint_init)
            self.carbon_components[
                "condition"] += self.gamma ** self.time_count * carbon_emissions * emissions_from_condition_perc
            self.carbon_components["actions"] += self.gamma ** self.time_count * carbon_emissions_from_actions
            self.carbon_components["total"] += self.gamma ** self.time_count * (
                        carbon_emissions - self.carbon_footprint_init + carbon_emissions_from_actions)


            # Calculate costs for carbon emissions
            step_cost[1] = (carbon_emissions - self.carbon_footprint_init) + carbon_emissions_from_actions

        if self.is_objective_active[2]:
            # Calculate the total travel time
            # step_cost[2] = - (np.sum(total_travel_time) - np.sum(self.TSTT_init))
            step_cost[2] = self._compute_user_cost(comp_traffic)
            self.convenience_components["user_cost"] += self.gamma ** self.time_count * step_cost[2]

            # Commenting out total travel time for now
            # self.convenience_components["travel_time"] -= self.gamma ** self.time_count * (
            #         np.sum(total_travel_time) - np.sum(self.TSTT_init))

        # Updating the ongoing actions
        self.act_ongoing = act_ongoing_tmp

        # Check if this was the last iteration
        if self.time_count == self.time.shape[1] - 1:
            done = True

        # Update the timestep
        self.time_count += 1

        # Prepare the new states that will be fed to the nn, by adding the normalized deterioration rates
        if self.use_cci_state:
            self.states_nn = np.concatenate(
                [self.states_cci.flatten(), self.states_iri.flatten(), self.time[:, self.time_count] / self.timesteps])
        else:
            self.states_nn = np.concatenate([self.states_iri.flatten(), [self.time_count]])

        # Log results
        if not self.quiet:
            print(f"Timestep: {self.time_count - 1}, Action: {action}, Cost: "
                  f"{self._normalize_rewards(step_cost)}")

        # Visualize the states of a component
        if self.plot_states:
            self._visualize_states(0, action, save=True)

        return self.states_nn, self._normalize_rewards(step_cost), done, \
            {"actions": action,
             "costs": step_cost,
             "states_iri": self.states_iri,
             # "traffic": comp_traffic,
             "urgent_components": self.urgent_comps,
             "cost_components": self.cost_components,
             "carbon_components": self.carbon_components,
             "convenience_components": self.convenience_components}

    def reset(self, *, seed=None, options=None, ):
        """
        Resets the environment in the initial state
        :return: -
        """
        if self.use_cci_state:
            self.states_cci = np.zeros((self.num_components, self.num_states_cci))
        self.states_iri = np.zeros((self.num_components, self.num_states_iri))

        # set the state types:
        if self.use_cci_state:
            self.state_types = [self.states_cci, self.states_iri]
        else:
            self.state_types = [self.states_iri]

        # Reset the state and other parameters, if needed
        for comp in range(self.num_components):
            self._determine_state(comp, detState=[0, 0])

        self.time_count = 0
        self.time = np.tile(np.arange(self.timesteps), (self.num_components, 1))

        if self.use_cci_state:
            self.states_nn = np.concatenate([self.states_cci.flatten(), self.states_iri.flatten(),
                                             self.time[:, self.time_count] / self.timesteps])
        else:
            self.states_nn = np.concatenate([self.states_iri.flatten(), [self.time_count]])

        for key, link in self.road_network.linkSet.items():
            self.road_network.linkSet[key].flow = self.road_network.linkSet[key].flow_init

        # Logging
        self.cost_components = {"maintenance": 0, "inspection": 0, "mobilization": 0, "urgent_actions": 0, 'total': 0}
        self.carbon_components = {"rerouting": 0, "condition": 0, "actions": 0, 'total': 0}
        self.convenience_components = {
            # "travel_time": 0,
            "user_cost": 0
        }

        return

    def render(self, mode='human', close=False):
        """
        Method not used for now.
        :param mode: -
        :param close: -
        :return:
        """
        pass

    def _load_yaml_file(self, filename):
        """
        Method that loads the parameter file.
        :return: dictionary containing all parameters
        """
        with open(filename, 'rb') as f:
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

    def _load_files(self, pattern="_filename"):
        """
        Iterates over all attributes and check which of them contain the pattern. It then,
        loads the files, the filenames of which are in the marked attributes.
        :param pattern: str - the pattern to look in the attributes
        :return:
        """
        all_attributes_dict = self.__dict__.copy()
        for key, value in all_attributes_dict.items():
            # Check if the
            if key.endswith(pattern):
                # check if file is a csv:
                if value.endswith(".txt") or value.endswith(".csv"):
                    file_values = np.loadtxt(value, delimiter=",")
                elif value.endswith(".npy"):
                    file_values = np.load(value)
                setattr(self, key.split(pattern)[0], file_values)

    def get_immediate_cost_risk(self, action):
        act_init = action
        is_component_active = np.ones(self.num_components, dtype=bool)
        cost_action = sum(self.c_mai[self.components, action])
        cost_ins = sum(self.gamma * self.c_ins[self.components, action])

        cost_action_crew = np.sum(self.c_mai_crew[np.unique(action)])
        cost_ins_crew = np.sum(self.c_ins_crew[np.unique(action)])

        cost_action += cost_action_crew
        cost_ins += cost_ins_crew

        cost_risk = np.sum(self.states_iri[:, -1] * 1.5 * self.c_mai[self.components, -1])

        return cost_action, cost_ins, is_component_active, act_init, action, cost_risk

    def get_immediate_cost(self, action):
        act_init = np.zeros(self.num_components, dtype=int)

        # Check if any component is in terminal state
        if self.use_cci_state:
            state_cci_inf = np.array([np.random.choice(range(self.num_states_cci), replace=True,
                                                       p=state_probs) for state_probs in self.states_cci])
            action[state_cci_inf == self.num_states_cci - 1] = 9  # replacement if CCI state is terminal

        state_iri_inf = np.array([np.random.choice(range(self.num_states_iri), replace=True,
                                                   p=state_probs) for state_probs in self.states_iri])
        action[state_iri_inf == self.num_states_iri - 1] = 9  # replacement if IRI state is terminal

        if self.use_cci_state:
            self.urgent_comps = np.where((state_iri_inf == self.num_states_iri - 1) | (state_cci_inf ==
                                                                                       self.num_states_cci - 1))[0]
        else:
            self.urgent_comps = np.where(state_iri_inf == self.num_states_iri - 1)[0]

        # Compute the cost of urgent actions. We assume that these actions cost 50% more
        cost_action_urgent = sum(self.c_mai[self.urgent_comps, action[self.urgent_comps]]) * 1.5
        cost_ins_urgent = self.gamma * sum(self.c_ins[self.urgent_comps, action[self.urgent_comps]]) * 1.5

        if self.limit_budget:  # TODO - Revisit budget limit after the Thesis!
            if self.budget_lim - self.episode_cost < 0:  # Check if we have budget remaining
                comp_active = []
            else:
                comp_active_cand = [i for i in range(self.num_components) if i not in np.where(
                    self.act_ongoing)[0]]
                comp_active_cand = [i for i in comp_active_cand if i not in self.urgent_comps]

                cost_action = sum(self.c_mai[comp_active_cand, action[comp_active_cand]])
                cost_ins = self.gamma * sum(self.c_ins[comp_active_cand, action[comp_active_cand]])

                step_cost_potential = - cost_action - cost_ins
                step_cost_potential_urgent = - cost_ins_urgent - cost_action_urgent
                comp_active = comp_active_cand

                if step_cost_potential > self.budget_lim - self.episode_cost - \
                        step_cost_potential_urgent:
                    comp_order = np.random.permutation(comp_active_cand)
                    step_cost_actual = cost_ins_urgent + cost_action_urgent
                    comp_active = []
                    for j in comp_order:
                        step_cost_actual -= self.c_mai[j][action[j]]
                        step_cost_actual -= self.gamma * self.c_ins[j][action[j]]
                        if step_cost_actual <= self.budget_lim - self.episode_cost:
                            comp_active += [j]
                        else:
                            step_cost_actual += self.c_mai[j][action[j]]
                            step_cost_actual += self.gamma * self.c_ins[j][action[j]]

                # action_real = np.zeros((1,tot_comp),dtype = int)
            act_init[comp_active + list(self.urgent_comps)] = action[comp_active + list(self.urgent_comps)]
            cost_ins = self.gamma * sum(self.c_ins[comp_active, action[comp_active]]) + \
                       self.gamma * sum(self.c_ins[self.urgent_comps, action[self.urgent_comps]])
            cost_action = sum(self.c_mai[comp_active, action[comp_active]]) + cost_action_urgent
            # cost_delay = sum(self.c_delay[comp_active,action[comp_active]])

            is_component_active = np.zeros(self.num_components, dtype=bool)
            is_component_active[comp_active + list(self.urgent_comps)] = True

        # If no budget limit is applied, compute the total actions cost
        else:
            is_component_active = np.ones(self.num_components, dtype=bool)
            cost_action = sum(self.c_mai[self.components, action]) + \
                          sum(self.c_mai[self.urgent_comps, action[self.urgent_comps]]) * 0.5
            cost_ins = self.gamma * sum(self.c_ins[self.components, action]) + \
                       self.gamma * sum(self.c_ins[self.urgent_comps, action[self.urgent_comps]]) * 0.5

        # Add cost for risk
        risk = 0  # No risk is included in this implementation

        # Add crew cost. Currently, the crew cost is the 50% of the mean action cost
        cost_action_crew = np.sum(self.c_mai_crew[np.unique(action)])
        cost_ins_crew = np.sum(self.c_ins_crew[np.unique(action)])

        # Log the costs
        self.cost_components["maintenance"] += self.gamma ** self.time_count * (cost_action - cost_action_urgent)
        self.cost_components["inspection"] += self.gamma ** self.time_count * cost_ins
        self.cost_components["mobilization"] += self.gamma ** self.time_count * (cost_action_crew + cost_ins_crew)
        self.cost_components["urgent_actions"] += self.gamma ** self.time_count * cost_action_urgent
        self.cost_components["total"] += self.gamma ** self.time_count * (
                    cost_action + cost_ins + cost_action_crew + cost_ins_crew)

        cost_action += cost_action_crew
        cost_ins += cost_ins_crew

        return cost_action, cost_ins, is_component_active, act_init, action, risk

    def compute_action_cost(self):
        """
        Compute the total cost for each action, as a combination of the maintenance and
        inspection cost
        :return:
        """
        c_ins_rep = np.repeat(self.c_ins, self.c_mai.shape[1], axis=1)
        c_act_rep = np.tile(self.c_mai, self.c_ins.shape[1])

        return c_ins_rep + c_act_rep

    def _update_states(self, comp, tp_dict):
        """
        Updates states by multiplying the old states with the transition probability matrix.
        :param comp: int - Component index
        :return:
        """
        if len(tp_dict["cci"].shape) == 4:  # only for tp_cci_dn
            tp_cci = tp_dict["cci"][:, :, self.time[comp, self.time_count], comp]
        else:
            tp_cci = tp_dict["cci"]
        tp_iri = tp_dict["iri"]

        if self.use_cci_state:
            self.states_cci[comp] = np.matmul(self.states_cci[comp], tp_cci)
            self.states_cci[comp] = np.clip(self.states_cci[comp], 0, None)

        self.states_iri[comp] = np.matmul(self.states_iri[comp], tp_iri)
        self.states_iri[comp] = np.clip(self.states_iri[comp], 0, None)
        self.states_iri[comp] /= np.sum(self.states_iri[comp])
        return

    def _shift_states(self, comp, shift=1):
        """
        Shifts the states distribution by a specified distance.
        :param comp: int - Component index
        :param shift: int - Distance for the shift
        :return: -
        """
        # When a repair action is performed, the states should shift back one place
        for states in self.state_types:
            states[comp] = np.roll(states[comp], -shift)
            states[comp, -1] = 0
            states[comp, 0] = 1 - np.sum(states[comp, 1:])
            states[comp] = np.clip(states[comp], 0, None)
        return

    def _determine_state(self, comp, detState=[0, 0]):
        """
        Determines state, in case the states distribution after the step is deterministic.
        :param comp: int - Component index.
        :param detState: int - State index in which the component is.
        :return: -
        """
        # When for some reason the state becomes known, update the states distribution accordingly
        # Default: reset to state 0
        for idx, states in enumerate(self.state_types):
            states[comp, :] = 0
            states[comp, detState[idx]] = 1
        return

    def _hold_deter_rate(self, comp, steps=5):
        """
        Preserves the same deterioration rate.
        :param comp: int - Component index
        :param steps: int - Number of steps for which the deterioration rate is preserved
        :return: -
        """
        # Hold the deterioration rate still for a predefined number of steps
        self.time[comp, self.time_count: self.time_count + steps] = self.time[
            comp, self.time_count]

        self.time[comp, self.time_count + steps:] = np.arange(
            len(self.time[comp, self.time_count + steps:])) + self.time[comp, self.time_count] + 1
        return

    def _reset_deter_rate(self, comp, cur_action):
        """
        Set the deterioration rate to the initial one
        :param comp: int - Component index
        :return: -
        """
        self.time[comp, self.time_count:] = np.arange(len(self.time[comp, self.time_count:]))
        # Set next year equal to zero, if the selected action takes more than 1 year
        if self.actions_long[comp, cur_action]:
            self.time[comp, min(self.time_count + 1, self.time.shape[1] - 1)] = 0
        return

    def _shift_deter_rate(self, comp, shift=1):
        """
        Shifts the states distribution by a specified distance.
        :param comp: int - Component index
        :param shift: int - Distance for the shift
        :return: -
        """
        # The deretioration rate goes back <shift> positions
        self.time[comp, self.time_count:] = np.clip(self.time[comp, self.time_count:] - shift, a_min=0, a_max=None)
        return

    def _infer_state(self, comp, obsState):
        """
        Infer the new states distribution after the inspection
        :param comp: int - component index
        :param obsState: int - observed state index
        :return: -
        """

        nominator = self.states[comp] * np.reshape(self.obs_probs[:, obsState], (self.num_states))
        self.states[comp] = nominator / sum(nominator)
        return

    def _normalize_trans_probs(self):
        """
        Normalize the transition probabilities, coming as input, so that they add-up to 1
        :return: -
        """
        for comp in range(self.tp.shape[0]):
            for i in range(self.tp.shape[1]):
                for j in range(self.tp.shape[2]):
                    self.tp[comp, i, j] /= np.sum(self.tp[comp, i, j])
        return

    def _construct_obs_prob_matrix(self):
        """
        Construct the observation matrix, given the probability of a correct observation
        :return:
        """
        self.obs_probs = np.zeros([self.num_states, self.num_states])
        pattern = [(1 - self.correct_prob_obs) / 2, self.correct_prob_obs,
                   (1 - self.correct_prob_obs) / 2]
        pattern_start = [self.correct_prob_obs, (1 - self.correct_prob_obs)]
        pattern_end = [(1 - self.correct_prob_obs), self.correct_prob_obs]
        for i in range(self.num_states):
            if i == 0:
                self.obs_probs[0, 0:2] = pattern_start
            elif i == self.num_states - 1:
                self.obs_probs[-1, -2:] = pattern_end
            else:
                self.obs_probs[i, i - 1:i + 2] = pattern

    def _visualize_states(self, comp, action, save=False):
        cur_action = action[comp]
        plt.bar(np.arange(self.num_states_iri), self.states_iri[comp])
        actions_dict = {}
        for key, value in self.action_results.items():
            actions_dict[key] = value["name"]
        plt.title(
            f"timestep {self.time_count - 1} - Action {cur_action} : {actions_dict[cur_action]}")
        plt.xlabel("CCI condition of component 1")
        plt.ylabel("Probability of each state")
        plt.ylim((0, 1))
        if save:
            plt.savefig(f"fig_{str(self.time_count).zfill(2)}.png")
        plt.show()

    def _traffic_assignment(self, action, is_comp_active):
        # Solve traffic assignment for each month. Check if the month-to-month maintenance
        #  changes
        TSTT = np.zeros(12)
        flows = np.zeros((self.num_components, 12))
        times = np.zeros((self.num_components, 12))
        ffts = np.zeros((self.num_components, 12))

        # Find the duration of all planned actions
        act_seq = action * is_comp_active
        act_durations = self.act_duration[tuple(range(self.num_components)), tuple(act_seq)]

        for idx in range(self.num_components):
            if self.act_ongoing[idx]:  # if an action is ongoing from previous year,
                # it is probably a replacement. Remove 365 days
                act_durations[idx] = self.act_duration[idx, -1] - 365

        # Execute traffic assignment for the first month
        month_intv = 30
        # act_ongoing_flag = act_durations >= month_intv*(0 + 0.5)

        # TSTT[0], flows[:, 0] = self._execute_traffic_assignment(act_ongoing_flag)

        import time
        months_to_check = [0, 6]
        act_ongoing_flag = np.zeros_like(act_durations, dtype=bool)
        for monthIdx in months_to_check:
            now = time.time()
            act_ongoing_flag_new = act_durations >= month_intv * (monthIdx + 0.5)
            if np.all(act_ongoing_flag_new == act_ongoing_flag) and monthIdx != 0:
                # print("Same network as previously")
                # TSTT[monthIdx] = TSTT[monthIdx - 1]
                # flows[:, monthIdx] = flows[:, monthIdx - 1]
                TSTT[monthIdx] = TSTT[months_to_check[months_to_check.index(monthIdx) - 1]]
                flows[:, monthIdx] = flows[:, months_to_check[months_to_check.index(monthIdx) - 1]]
            elif act_ongoing_flag_new.tobytes() in self.traffic_assignment_solutions.keys():
                # print("Load solution from buffer")
                TSTT[monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][0]
                flows[:, monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][1]
                times[:, monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][2]
                ffts[:, monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][3]
            else:
                # print("Solving network from scratch")
                TSTT[monthIdx], flows[:, monthIdx], times[:, monthIdx], ffts[:, monthIdx] = self._execute_traffic_assignment(act_ongoing_flag_new)
                self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()] = [TSTT[monthIdx], flows[:, monthIdx],times[:, monthIdx], ffts[:, monthIdx]]

            # print(f"Time to solve the network: {time.time() - now}")
            act_ongoing_flag = act_ongoing_flag_new

        TSTT[:] = np.mean(TSTT[months_to_check])
        flows = np.tile(np.mean(flows[:, months_to_check], axis=1), (12, 1)).T
        times = np.tile(np.mean(times[:, months_to_check], axis=1), (12, 1)).T
        ffts = np.tile(np.mean(ffts[:, months_to_check], axis=1), (12, 1)).T
        # flows = np.mean(flows[:, [0,5]], axis=1)
        return TSTT, flows, times, ffts

    def _execute_traffic_assignment(self, act_ongoing_flag):

        ongoing_actions = np.array(self.components)[np.where(act_ongoing_flag == 1)[0]]
        for key, link in self.road_network.linkSet.items():
            # self.road_network.linkSet[key].capacity = self.road_network.linkSet[
            #     key].max_capacity

            if link.comp_idx in ongoing_actions:
                self.road_network.linkSet[key].capacity = self.road_network.linkSet[key].max_capacity_init * \
                                                          self.closure_perc
            else:
                self.road_network.linkSet[key].capacity = self.road_network.linkSet[
                    key].max_capacity_init

            self.road_network.linkSet[key].max_capacity = self.road_network.linkSet[key].capacity

            # self.road_network.linkSet[key].flow = self.road_network.linkSet[key].capacity / 2
            # self.road_network.linkSet[key].flow = 0

            # Reinitialize the flow of the link with the "Do Nothing" flow
            self.road_network.linkSet[key].flow = self.road_network.linkSet[key].flow_init

        TSTT = assignment_loop(
            network=self.road_network, algorithm="FW",
            systemOptimal=False,
            costFunction=BPRcostFunction,
            accuracy=0.02, maxIter=200, maxTime=6000000, verbose=False)

        self.num_traffic_assignments += 1

        if TSTT <= 0:
            TSTT = len(self.road_network.linkSet) * len(self.road_network.tripSet) * np.mean(
                self.comp_len[self.components]) * 60 * 1e2 * 12
            flow = np.ones(self.num_components) * len(self.road_network.tripSet) * 1e6 * (
                    act_ongoing_flag == 0) * 12
            times = np.ones(self.num_components) * len(self.road_network.tripSet) * 1e6 * (
                    act_ongoing_flag == 0) * 12
            ffts = np.ones(self.num_components) * len(self.road_network.tripSet) * 1e6 * (
                    act_ongoing_flag == 0) * 12
        else:
            flow = np.zeros(self.num_components)
            times = np.zeros(self.num_components)
            ffts = np.zeros(self.num_components)
            for idx, x in enumerate(self.road_network.linkSet.keys()):
                flow[idx] = self.road_network.linkSet[x].flow
                times[idx] = self.road_network.linkSet[x].cost
                ffts[idx] = self.road_network.linkSet[x].fft

        return TSTT * 30 * 24, flow * 30 * 24, times, ffts  # Multiply x30 to get the monthly trips and travel time

    def _compute_carbon_footprint(self, comp_traffic, comp_time, comp_fft):
        # Compute the average carbon footprint based on the CO2e emissions and vehicle types
        roughness_per_segment = np.array(self.iri_values)[np.argmax(self.states_iri, axis=1)]
        extra_consumption_per_segment_per_type = (np.reshape(roughness_per_segment, (-1, 1)) @ np.reshape(
            self.carbon_emissions_iri_a, (1, -1)) + np.tile(self.carbon_emissions_iri_b,
                                                            (self.num_components, 1))) / 100
        extra_consumption_per_segment_per_type = np.clip(extra_consumption_per_segment_per_type, a_min=0,
                                                         a_max=None) + 1
        average_emissions_per_segment = extra_consumption_per_segment_per_type @ (
                    np.asarray(self.transport_types) * np.asarray(self.carbon_foot_by_type))

        emissions_from_condition_perc = np.dot(
            np.dot(extra_consumption_per_segment_per_type, self.transport_types),
            self.comp_len[self.components]) / np.sum(self.comp_len[self.components]) - 1

        time_ratio = comp_time / comp_fft
        # time_ratio = 1

        return - np.sum(self.comp_len[self.components] * np.sum(comp_traffic * time_ratio, axis=1) *
                        average_emissions_per_segment), \
            emissions_from_condition_perc

    def _compute_carbon_footprint_actions(self, action):
        return -np.sum(np.array(self.action_carbon_emissions)[action] * self.comp_len[self.components]) * 1000 * 3.7 * 2

    def _compute_user_cost(self, comp_traffic):

        # roughness_per_segment = np.array(self.iri_values)[np.argmax(self.states_iri, axis=1)]
        # roughness_per_segment_per_means_trans = np.repeat([roughness_per_segment], 5, axis=0)
        # roughness_per_segment_trans = np.multiply(np.array(self.transport_types)[:, np.newaxis], roughness_per_segment_per_means_trans)


        # Calculate the user cost per IRI state, per transport type
        IRI = np.reshape(self.iri_values, (1, -1))
        user_cost_per_state_vt = np.reshape(self.a0, (-1, 1)) + \
                            np.reshape(self.a1, (-1, 1)) @ IRI + \
                            np.reshape(self.a2, (-1, 1)) @ IRI ** 2 + \
                            np.reshape(self.a3, (-1, 1)) @ IRI ** 3

        # Compute the user cost per IRI state
        user_cost_per_state = np.array(self.transport_types) @ user_cost_per_state_vt

        # Compute the user cost per segment (in $/per vehicle km)
        user_cost_per_segment = user_cost_per_state[np.argmax(self.states_iri, axis=1)]

        return -np.sum(user_cost_per_segment * np.sum(comp_traffic, axis=1) * self.comp_len[self.components])

    def _filter_components_actions(self):
        # Filter specific components
        if self.components_to_keep:
            self.num_components = len(self.components_to_keep)
            self.components = self.components_to_keep
        else:
            self.components = np.arange(self.num_components).tolist()

        # Filter specific actions
        if self.actions_to_remove:
            self.num_actions = self.num_maintenance * self.num_inspections - len(self.actions_to_remove)
            self.actions = np.array([x for x in range(self.num_maintenance * self.num_inspections)
                                     if x not in self.actions_to_remove])
        else:
            self.num_actions = self.num_maintenance * self.num_inspections
            self.actions = np.arange(self.num_actions)

    def _filter_road_network_components(self):
        road_network_tmp = deepcopy(self.road_network)

        # Remove the unused components from links
        for key, link in copy(road_network_tmp.linkSet).items():
            if link.comp_idx not in self.components:
                del road_network_tmp.linkSet[key]
                road_network_tmp.nodeSet[key[0]].outLinks.remove(key[1])
                road_network_tmp.nodeSet[key[1]].inLinks.remove(key[0])

        # Remove the unused nodes
        del_nodes = []
        for key, node in copy(road_network_tmp.nodeSet).items():
            if (len(node.inLinks) == 0) and (len(node.outLinks) == 0):
                del_nodes.append(key)
                del road_network_tmp.nodeSet[key]
                road_network_tmp.originZones.remove(key)
                del road_network_tmp.zoneSet[key]

        # Remove the unused tripSets
        for key, trip in copy(road_network_tmp.tripSet).items():
            if (key[0] in del_nodes) or (key[1] in del_nodes):
                del road_network_tmp.tripSet[key]

        # Remove the unused nodes from destination lists
        for key, trip in copy(road_network_tmp.zoneSet).items():
            # for node in road_network_tmp.zoneSet[key].destList:
            road_network_tmp.zoneSet[key].destList = list(i for i in
                                                          road_network_tmp.zoneSet[
                                                              key].destList if i not in del_nodes)
            # if (node in del_nodes):
            #     road_network_tmp.zoneSet[key].destList.remove(node)

        self.road_network = road_network_tmp

    @staticmethod
    def _perform_inference(cur_action_fin, states_dist, obs_dist):
        obs_dist_new = states_dist @ obs_dist[cur_action_fin]
        obs_dist_new = obs_dist_new / np.sum(obs_dist_new)
        obs = np.random.choice(range(states_dist.shape[0]), size=None, replace=True, p=obs_dist_new)

        update_dist = states_dist * obs_dist[cur_action_fin, :, obs]
        return update_dist / np.sum(update_dist)

    def _normalize_rewards(self, rewards):
        if self.norm_method == "standard":
            return (rewards - self.max_reward) / self.std_reward
        elif self.norm_method == "scaled":
            return rewards / self.norm_factor
        elif self.norm_method == "none":
            return rewards
        else:
            raise ValueError("Unknown normalization method")






if __name__ == "__main__":
    import logging

    logging.disable(logging.WARNING)
    os.chdir("../../../..")

    episodes = 10
    env = ThesisEnv()
    env.quiet = True
    results = []

    for ep in range(episodes):

        # Sample policy check
        all_costs = []
        all_states = []

        # Initialize tables for visualization
        actions = []
        costs = []
        # states_cci = []
        states_iri = []
        traffic = []
        episode_cost = np.zeros(3)

        tau = 0
        import time

        env.reset()
        env.num_traffic_assignments = 0
        begin_time = time.time()
        for i in range(1, 21):

            step_time = time.time()
            cur_action = 0

            # CBM best setting
            cur_action = [0] * env.num_components
            for comp in range(env.num_components):
                mean_state_iri = np.random.choice(range(env.states_iri.shape[1]), 1,
                                                  p=env.states_iri[comp])
                mean_state = np.max([mean_state_iri])
                if mean_state >= 5:
                    cur_action[comp] = 4
                elif mean_state >= 2:
                    cur_action[comp] = 1
                if (i % 1 == 0) and (cur_action[comp] != 4):
                    cur_action[comp] += 2

            states, step_cost, done, metadata = env.step(cur_action)

            # VISUALIZATIONS
            # actions.append(metadata["actions"])
            # costs.append(metadata["costs"])
            # # states_cci.append(metadata["states_cci"])
            # states_iri.append(metadata["states_iri"])
            # traffic.append(metadata["traffic"])
        results.append([ep, time.time() - begin_time, env.num_traffic_assignments])
        print(f"Episode {ep}: {time.time() - begin_time}, {env.num_traffic_assignments}")
        # print(f"Total reward is: {episode_cost}")
        #
        # np.save("actions_env_case4.npy", actions)
        # np.save("costs_env_case4.npy", costs)
        # # np.save("states_cci_env.npy", states_cci)
        # np.save("states_iri_env_case4.npy", states_iri)
        # np.save("traffic_env_case4.npy", traffic)
    # np.savetxt("timing_zero_each_time.csv", results, delimiter=",")
    # np.savetxt("timing_reset_to_init_problem.csv", results, delimiter=",")
    # np.savetxt("timing_leave_previous.csv", results, delimiter=",")
    # np.savetxt("timing_half_capacity.csv", results, delimiter=",")
    # np.savetxt("timing_beta_4.csv", results, delimiter=",")
    # np.savetxt("timing_beta_3.csv", results, delimiter=",")
    # np.savetxt("timing_beta_2.csv", results, delimiter=",")
