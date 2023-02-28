import gym
from gym import spaces
import yaml
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import copy, deepcopy

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

        # self._construct_obs_prob_matrix()

        # Set the seed for reproducibility
        if self.seed:
            np.random.seed(self.seed)

        # Filter specific components
        if self.components_to_keep:
            self.num_components = len(self.components_to_keep)
            self.components = self.components_to_keep
        else:
            self.components = np.arange(self.num_components).tolist()

        # Filter specific actions
        if self.actions_to_remove:
            self.num_actions = self.num_maintenance * self.num_inspections - len(
                self.actions_to_remove)
            self.actions = np.array([x for x in range(self.num_maintenance * self.num_inspections)
                            if x not in self.actions_to_remove])
        else:
            self.num_actions = self.num_maintenance * self.num_inspections
            self.actions = np.arange(self.num_actions)

        # Define the actions space
        self.action_space = spaces.MultiDiscrete([(self.num_actions)]*self.num_components)

        # Define the state space
        self.observation_space = spaces.MultiDiscrete([self.num_states_iri]*self.num_components)

        # Compute the total cost per action
        self.c_act = self.c_mai + self.c_ins
        # self.c_act = self.compute_action_cost()

        self.transitions = {
            "plain": {"cci": self.tp_cci_plain, "iri": self.tp_iri_plain},
            "minor": {"cci": self.tp_cci_minor, "iri": self.tp_iri_minor},
            "major": {"cci": self.tp_cci_major, "iri": self.tp_iri_major},
            "replace": {"cci": self.tp_cci_replace, "iri": self.tp_iri_replace}
        }

        self.act_ongoing = np.zeros(self.num_components, dtype=bool)

        self.episode_cost = 0

        # Initialize road as graph network
        self.net_file = "data/init_data/sioux_falls/SiouxFalls_net.csv"
        self.road_network = load_network(net_file=self.net_file)
        # For debugging purposes, to play with the capacity
        # for link in self.road_network.linkSet:
        #     self.road_network.linkSet[link].capacity = self.road_network.linkSet[link].capacity

        # Initialize buffer to keep traffic assignment solutions
        self.traffic_assignment_solutions = {}

        self.num_traffic_assignments = 0

        # If we only kept some of the network components, remove the rest
        if self.components_to_keep:
            road_network_tmp = deepcopy(self.road_network)
            for key, link in copy(road_network_tmp.linkSet).items():
                if link.comp_idx not in self.components:
                    del road_network_tmp.linkSet[key]
                    road_network_tmp.nodeSet[key[0]].outLinks.remove(key[1])
                    road_network_tmp.nodeSet[key[1]].inLinks.remove(key[0])

            del_nodes = []
            for key, node in copy(road_network_tmp.nodeSet).items():
                if (len(node.inLinks) == 0) and (len(node.outLinks) == 0):
                    del_nodes.append(key)
                    del road_network_tmp.nodeSet[key]
                    road_network_tmp.originZones.remove(key)
                    del road_network_tmp.zoneSet[key]

            # self.road_network = deepcopy(road_network_tmp)
            for key, trip in copy(road_network_tmp.tripSet).items():
                if (key[0] in del_nodes) or (key[1] in del_nodes):
                    del road_network_tmp.tripSet[key]

            for key, trip in copy(road_network_tmp.zoneSet).items():
                for node in road_network_tmp.zoneSet[key].destList:
                    road_network_tmp.zoneSet[key].destList = list(i for i in
                                                              road_network_tmp.zoneSet[
                                                                  key].destList if i not in del_nodes)
                    # if (node in del_nodes):
                    #     road_network_tmp.zoneSet[key].destList.remove(node)


            self.road_network = road_network_tmp

        for key, link in self.road_network.linkSet.items():
            # self.road_network.linkSet[key].beta = 2
            self.road_network.linkSet[key].max_capacity *= 0.25
            self.road_network.linkSet[key].capacity *= 0.25
            self.road_network.linkSet[key].flow_init = 0
            self.road_network.linkSet[key].max_capacity_init = self.road_network.linkSet[key].max_capacity

        # Reset the environment
        self.reset()
        self.time_count = 0

        self.TSTT_init, flows_init = self._execute_traffic_assignment(np.zeros_like(self.components, dtype=bool))

        self.carbon_footprint_init = self._compute_carbon_footprint(np.repeat([flows_init], repeats=12, axis=0).T)

        # self.TSTT_init = assignment_loop(
        #     network=self.road_network, algorithm="FW",
        #     systemOptimal=False,
        #     costFunction=BPRcostFunction,
        #     accuracy=0.02, maxIter=200, maxTime=6000000, verbose=False)

        for key, link in self.road_network.linkSet.items():
            self.road_network.linkSet[key].flow_init = self.road_network.linkSet[key].flow

        # Assign normalizing factors if they are not set
        if not self.norm_factor:
            self.norm_factor = np.zeros(self.num_objectives)
            self.norm_factor[0] = np.sum(np.max(np.abs(self.c_mai + self.c_ins)[self.actions],
                                                axis=1))/5
            avg_carbon = np.sum(np.asarray(self.transport_types) *  np.asarray(
                self.carbon_foot_by_type))
            self.norm_factor[1] = np.sum(self.comp_len[self.components] * self.capacity[
                self.components]) * avg_carbon * 12
            self.norm_factor[2] = np.sum(self.comp_len[self.components] * self.capacity[
                self.components]) * 10 * 12
        else:
            self.norm_factor = np.asarray(self.norm_factor)

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

        # maintenance_flag = np.zeros(self.num_maintenance, dtype=bool)
        # inspection_flag = np.zeros(self.num_inspections, dtype=bool)

        # Calculate the immediate costs
        cost_action, cost_insp, is_comp_active, act_real, action, cost_risk = self.get_immediate_cost(
            action)

        # Initialize the tmp variable for ongoing actions
        act_ongoing_tmp = self.act_ongoing.copy()

        # Iterate over components. We assume that maintenance and inspection takes place at the
        # beginning of the timestep.
        for idx, comp in enumerate(self.components):

            # Get current action
            cur_action = action[idx]

            # if idx == 5 and self.time_count >= 4:
            #     print(f"Start debugging for component {idx} and action {cur_action}")

            # Calculate the new state distribution and the cost, based on the action given

            # If we are in an ongoing action, reset the "ongoing_action" flag to False for the
            # next step
            if self.act_ongoing[idx]:
                act_ongoing_tmp[idx] = False
            elif (not self.act_ongoing[idx]) and (self.actions_long[idx, cur_action]) and (
                    is_comp_active[idx]):
                act_ongoing_tmp[idx] = True

            # If the component is active, update the deterioration rate and the state based on the
            # action
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
                cur_action = 0 # if the component is not active, assume that the action is "do
                # nothing"

            # Perform inference (belief update) if:
            # - There is no ongoing action
            # - We aren't starting an ongoing action
            if (not self.act_ongoing[idx]) and (not act_ongoing_tmp[idx]):
                if is_comp_active[idx]:
                    cur_action_fin = cur_action
                else:
                    cur_action_fin = 0 # If component is not active, then perform update with do nothing

                obs_dist_iri = self.states_iri[idx] @ self.obs_probs_iri[idx, cur_action_fin]
                obs_dist_iri = obs_dist_iri/np.sum(obs_dist_iri)
                obs_iri = np.random.choice(range(self.num_states_iri), size=None, replace=True, p=obs_dist_iri)

                update_iri = self.states_iri[idx] * self.obs_probs_iri[idx, cur_action_fin, :, obs_iri]
                self.states_iri[idx] = update_iri / np.sum(update_iri)

                if self.use_cci_state:
                    obs_dist_cci = self.states_cci[idx] @ self.obs_probs_cci[idx, cur_action_fin]
                    obs_dist_cci = obs_dist_cci/np.sum(obs_dist_cci)
                    obs_cci = np.random.choice(range(self.num_states_cci), size=None, replace=True,
                                               p=obs_dist_cci)

                    update_cci = self.states_cci[idx] * self.obs_probs_cci[idx, cur_action_fin, :,
                                                        obs_cci]
                    self.states_cci[idx] = update_cci / np.sum(update_cci)



        # For the whole system
        # Add up the costs from actions
        step_cost[0] = cost_action + cost_insp + cost_risk

        # Calculate the traffic flows in the network per month
        # comp traffic can be either (components x months)
        total_travel_time, comp_traffic = self._traffic_assignment(action, is_comp_active)

        # Visualize the road network
        if self.plot_road_network:
            node_ids_in_use = []
            for idx, coord in enumerate(self.node_coords):
                if coord[0] in self.edges[self.components]:
                    node_ids_in_use.append(idx)
            # plot_graph(np.asarray(self.node_coords)[node_ids_in_use], self.edges[self.components],
            #            comp_traffic[:, 0], title=f"Timestep {self.time_count}, "
            #                                      f"Month {'January'}, "
            #                                      f"closed segments "
            #                                      f"{np.where(act_real != 0)[0]}", min_max=(0,
            #                                                                                np.max(comp_traffic[:, 0])))

            # # Plot IRI states
            plot_graph(np.asarray(self.node_coords)[node_ids_in_use], self.edges[self.components],
                       # np.argmax(self.states_iri, axis=1),
                       np.zeros(len(self.components)),
                       title=f"IRI States - Timestep"
                                                                 f" {self.time_count}, "
                                                            f"Month {'January'}"
                                                            # f"{np.where(act_real != 0)[0]}"
                       , min_max=(0, 5))
            #
            #
            # # Plot actions
            # plot_graph(np.asarray(self.node_coords)[node_ids_in_use], self.edges[self.components],
            #            action, title=f"Actions - Timestep {self.time_count}, "
            #                                      f"Month {'January'}", min_max=(0, 9))


        # # Calculate the total carbon footprint for this step
        step_cost[1] = - (self._compute_carbon_footprint(comp_traffic) - self.carbon_footprint_init)

        # Calculate the total travel time
        step_cost[2] = - (np.sum(total_travel_time) - self.TSTT_init*12)

        # Updating the ongoing actions
        self.act_ongoing = act_ongoing_tmp

        # Check if this was the last iteration
        if self.time_count == self.time.shape[1] - 1:
            done = True

        # Prepare the new states that will be fed to the nn, by adding the normalized deterioration rates
        if self.use_cci_state:
            self.states_nn = np.concatenate(
                [self.states_cci.flatten(), self.states_iri.flatten(),  self.time[:, self.time_count]
                 / self.timesteps])
        else:
            self.states_nn = np.concatenate([self.states_iri.flatten()])

        # Update the timestep
        self.time_count += 1

        # Log results
        if not self.quiet:
            print(f"Timestep: {self.time_count - 1}, Action: {action}, Cost: "
                  f"{step_cost / self.norm_factor}")

        # Visualize the states of a component
        if self.plot_states:
            self._visualize_states(0, action, save=True)

        return self.states_nn, step_cost / self.norm_factor, done, \
                                                                    {"actions": action,
                                                                     "costs": step_cost,
                                                                     "states_iri": self.states_iri,
                                                                     # "traffic": comp_traffic,
                                                                     "urgent_components": self.urgent_comps}

    # def reset(self):
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
            self.states_nn = np.concatenate([self.states_iri.flatten()])


        for key, link in self.road_network.linkSet.items():
            self.road_network.linkSet[key].flow = self.road_network.linkSet[key].flow_init
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
        act_real = action
        is_component_active = np.ones(self.num_components, dtype=bool)
        cost_action = sum(self.c_mai[self.components, action])
        cost_insp = sum(self.gamma*self.c_ins[self.components, action])

        cost_risk = np.sum(self.states_iri[:, -1] * 1.5 * self.c_mai[self.components, -1])

        return cost_action, cost_insp, is_component_active, act_real, action, cost_risk

    def get_immediate_cost(self, action):
        act_real = np.zeros(self.num_components, dtype=int)

        # Check if any component is in terminal state
        if self.use_cci_state:
            state_cci_inf = np.array([np.random.choice(range(self.num_states_cci), replace=True,
                                                 p=state_probs) for state_probs in self.states_cci])
            action[state_cci_inf == self.num_states_cci - 1] = 9 # replacement if CCI state is terminal

        state_iri_inf = np.array([np.random.choice(range(self.num_states_iri), replace=True,
                                             p=state_probs) for state_probs in self.states_iri])
        action[state_iri_inf == self.num_states_iri - 1] = 9 # replacement if IRI state is terminal

        if self.use_cci_state:
            self.urgent_comps = np.where((state_iri_inf == self.num_states_iri - 1) | (state_cci_inf ==
                                                                         self.num_states_cci - 1))[0]
        else:
            self.urgent_comps = np.where(state_iri_inf == self.num_states_iri - 1)[0]

        cost_action_urgent = sum(self.c_mai[self.urgent_comps,action[self.urgent_comps]])*1.5 # Assume that
        # urgent actions cost 50% more
        cost_ins_urgent = self.gamma*sum(self.c_ins[self.urgent_comps,action[self.urgent_comps]])*1.5 # Assume that
        # urgent actions cost 50% more

        if self.limit_budget:
            if self.budget_lim - self.episode_cost < 0: # Check if we have budget remaining
                comp_active = []
            else:
                comp_active_cand = [i for i in range(self.num_components) if i not in np.where(
                    self.act_ongoing)[0]]
                comp_active_cand = [i for i in comp_active_cand if i not in self.urgent_comps]

                cost_action = sum(self.c_mai[comp_active_cand,action[comp_active_cand]])
                cost_ins = self.gamma*sum(self.c_ins[comp_active_cand,action[comp_active_cand]])


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

                #action_real = np.zeros((1,tot_comp),dtype = int)
            act_real[comp_active + list(self.urgent_comps)] = action[comp_active + list(self.urgent_comps)]
            cost_ins = self.gamma*sum(self.c_ins[comp_active,action[comp_active]]) + \
                       self.gamma*sum(self.c_ins[self.urgent_comps,action[self.urgent_comps]])
            cost_action = sum(self.c_mai[comp_active,action[comp_active]]) + cost_action_urgent
            # cost_delay = sum(self.c_delay[comp_active,action[comp_active]])

            is_component_active = np.zeros(self.num_components, dtype=bool)
            is_component_active[comp_active + list(self.urgent_comps)] = True

        # If no budget limit is applied
        else:
            act_real = action
            is_component_active = np.ones(self.num_components, dtype=bool)
            cost_action = sum(self.c_mai[self.components, action]) + \
                          sum(self.c_mai[self.urgent_comps,action[self.urgent_comps]]) * 0.5
            cost_ins = self.gamma*sum(self.c_ins[self.components, action]) + \
                       self.gamma * sum(self.c_ins[self.urgent_comps, action[self.urgent_comps]]) * 0.5

        # Add cost for risk
        risk = 0 # No risk is included in this implementation

        # Add crew cost. Currently, the crew cost is the 50% of the mean action cost
        cost_action_crew = np.sum(self.c_mai_crew[np.unique(action)])
        cost_ins_crew = np.sum(self.c_ins_crew[np.unique(action)])

        cost_action += cost_action_crew
        cost_ins += cost_ins_crew

        return cost_action, cost_ins, is_component_active, act_real, action, risk

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
        if len(tp_dict["cci"].shape) == 4: # only for tp_cci_dn
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

        # Find the duration of all planned actions
        act_seq = action * is_comp_active
        act_durations = self.act_duration[tuple(range(self.num_components)), tuple(act_seq)]

        for idx in range(self.num_components):
            if self.act_ongoing[idx]: # if an action is ongoing from previous year,
                # it is probably a replacement. Remove 365 days
                act_durations[idx] = self.act_duration[idx, -1] - 365

        # Execute traffic assignment for the first month
        month_intv = 30
        act_ongoing_flag = act_durations >= month_intv*(0 + 0.3)

        TSTT[0], flows[:, 0] = self._execute_traffic_assignment(act_ongoing_flag)

        import time

        for monthIdx in range(1, 12):
            now = time.time()
            act_ongoing_flag_new = act_durations >= month_intv*(monthIdx + 0.3)
            if np.all(act_ongoing_flag_new == act_ongoing_flag):
                # print("Same network as previously")
                TSTT[monthIdx] = TSTT[monthIdx - 1]
                flows[:, monthIdx] = flows[:, monthIdx - 1]
            elif act_ongoing_flag_new.tobytes() in self.traffic_assignment_solutions.keys():
                # print("Load solution from buffer")
                TSTT[monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][0]
                flows[:, monthIdx] = self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()][1]
            else:
                # print("Solving network from scratch")
                TSTT[monthIdx], flows[:, monthIdx] = self._execute_traffic_assignment(act_ongoing_flag_new)
                self.traffic_assignment_solutions[act_ongoing_flag_new.tobytes()] = [TSTT[monthIdx], flows[:, monthIdx]]

            # print(f"Time to solve the network: {time.time() - now}")
            act_ongoing_flag = act_ongoing_flag_new

        return TSTT, flows

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
            TSTT = len(self.road_network.linkSet)*len(self.road_network.tripSet)*np.mean(
                self.comp_len[self.components])*60*1e2*12
            flow = np.ones(self.num_components)*len(self.road_network.tripSet)*1e6*(
                    act_ongoing_flag == 0)*12
        else:
            flow = np.zeros(self.num_components)
            for idx, x in enumerate(self.road_network.linkSet.keys()):
                flow[idx] = self.road_network.linkSet[x].flow

        return TSTT, flow

    def _compute_carbon_footprint(self, comp_traffic):
        # Compute the average carbon footprint based on the CO2e emissions and vehicle types
        roughness_per_segment = np.array(self.iri_values)[np.argmax(self.states_iri, axis=1)]
        extra_consumption_per_segment_per_type = np.reshape(roughness_per_segment, (-1, 1)) @ np.reshape(self.carbon_emissions_iri_a, (1, -1)) + np.tile(self.carbon_emissions_iri_b, (self.num_components, 1))
        extra_consumption_per_segment_per_type = np.clip(extra_consumption_per_segment_per_type,
                                                         a_min=0, a_max=None)
        average_emissions_per_segment = extra_consumption_per_segment_per_type @ (np.asarray(self.transport_types) * np.asarray(self.carbon_foot_by_type))

        return np.sum(self.comp_len[self.components] * np.sum(comp_traffic, axis=1) *
                      average_emissions_per_segment)

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
        env.num_traffic_assignments=0
        begin_time = time.time()
        for i in range(1, 21):
        # for i in range(1, 4):


            step_time = time.time()
            cur_action = 0
            # if i % 2 == 0: # minor repair
            #     cur_action = 9
            # elif i % 8 == 0: # major repair
            #     cur_action = 2
            # elif i % 18 == 0: # replace
            #     cur_action = 9
            # if i % 4 == 0 and cur_action != 9: # low fidelity inspection
            #     cur_action += 3
            # elif i % 7 == 0 and cur_action != 9: # high fidelity inspection
            #     cur_action += 6
            # cur_action = np.random.choice(env.actions, size=env.num_components, replace=True,
            #                            p=[0.8] + [0.2/(env.num_actions-1)]*(env.num_actions-1)
            #                               )
            # cur_action = np.zeros(env.num_components, dtype=int)

            # When actions are being taken from the NN, filter them like: env.actions[actions] TODO
            # print(f"Current action is {cur_action}")
            # states, step_cost, done, metadata = env.step(env.actions[cur_action])

            # Case 1: Do nothing
            # cur_action = np.zeros(env.num_components, dtype=int)

            # Case 2: Always repair
            # cur_action = np.ones(env.num_components, dtype=int)*7

            # Case 3: Replace every 5 years
            # if i % 5 == 0:
            #     cur_action = np.ones(env.num_components, dtype=int)*9
            # else:
            #     cur_action = np.ones(env.num_components, dtype=int)*0
            #
            # # Case 4: Replace every 5 years
            # if i % 6 == 0:
            #     cur_action = np.ones(env.num_components, dtype=int)*9
            # else:
            #     cur_action = np.ones(env.num_components, dtype=int)*0

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
                # elif mean_state >= minor_repair_thres:
                #     action[comp] = 1
                if (i % 1 == 0) and (cur_action[comp] != 4):
                    cur_action[comp] += 2

            states, step_cost, done, metadata = env.step(cur_action)
            # states, step_cost, done, _ = env.step(np.array([cur_action] * env.num_components))
            # print(f"Step time is {time.time() - step_time}")
            # if i % 5 == 0:
            #     tau = 0
            #     env.episode_cost = 0
            # else:
            #     env.episode_cost += env.gamma** tau * step_cost[0]*env.norm_factor[0]
            #     tau += 1

            # all_states.append(states)
            # all_costs.append(step_cost)
            #
            # episode_cost = episode_cost + step_cost

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
