import numpy as np
import os
import thesis_env
import gym
from datetime import datetime
from tqdm import tqdm
import yaml

from PPO import MindmapPPO


class Benchmarks:
    """
    Class that contains all the code benchmarks.
    """

    def __init__(self, param_file="src/benchmarks_params.yaml"):
        self.param_file = param_file
        self.param_dict = self._load_yaml_file(self.param_file)

        self.seed = self.param_dict["seed"]
        self.iters = self.param_dict["iters"]
        self.save = self.param_dict["save"]
        self.modes = self.param_dict["modes"]
        self.env_name = self.param_dict["env_name"]
        self.env = gym.make(self.env_name, quiet=True)

        # Set random seed for reproducibility
        if self.seed:
            np.random.seed(self.seed)

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    def execute_benchmarks(self):
        """
        This method serves as a wrapper to execute the benchmarks multiple times
        """

        for mode in self.modes:
            self.output_dir = f"outputs/benchmarks/{mode}_{self.timestamp}/"

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            #
            f = open(f"{self.output_dir}benchmark_costs.txt", "w")
            #
            inspect_res, minor_repair_res, major_repair_res, replace_res = \
                self.perform_experiment(mode=mode)

            # Write on results file
            f.write(f"Results for best {mode} benchmark\n")
            f.write(f"inspect int: {inspect_res}, minor_repair res: {minor_repair_res}, "
                    f"major_repair res: {major_repair_res},"
                    f"replace res: "
                    f"{replace_res}\n")
            list_costs_scalarized = []
            list_costs = []
            for i in range(1000):
                cost_scalarized, cost = eval(f"self.{mode}_based_rule(inspect_res, minor_repair_res, "
                            f"major_repair_res, "
                            f"replace_res)")
                list_costs_scalarized.append(cost_scalarized)
                list_costs.append(cost)
                f.write(f"Iteration {i + 1}: Scalarized Cost {cost_scalarized}\n")
            avg_cost_scalarized = np.mean(list_costs_scalarized, axis=0)
            avg_cost = np.mean(list_costs, axis=0)
            f.write(f"Average scalarized cost: {avg_cost_scalarized}")
            f.write(f"Average cost: {avg_cost}")
            print(f"Average scalarized cost: {avg_cost_scalarized}")
            f.write(f"Average cost: {avg_cost}")
            f.close()

        return avg_cost_scalarized, avg_cost

    def perform_experiment(self, mode):
        """
        Performs an experiment of specified length and a specified mode.
        :param mode: str - "time" and "condition"
        :return: -
        """
        print(f"Performing experiment in {mode} mode for {self.iters} iterations.")
        experiments = []

        for num_iter in range(self.iters):
            print(f"============ Iteration {num_iter + 1} ========================")
            if mode == "time":
                experiment_cost = self.time_based_benchmark(num_iter, mode)
            elif mode == "condition":
                experiment_cost = self.condition_based_benchmark(num_iter, mode)
            else:
                raise "Experiment should either be in 'time' or 'condition' mode"
            experiments.append(experiment_cost)

            if self.save:
                output_name = f"experiment_{self.timestamp}_iter_{num_iter}"
                np.save(self.output_dir + output_name, experiment_cost)

        # Find the overall best
        print(f"============ Finding the overall best combination ========================")
        experiments_np = np.stack(experiments)
        experiments_avg = np.mean(experiments_np, axis=0)

        inspect_res, minor_repair_res, major_repair_res, replace_res = self.calculate_min(
            experiments_avg, mode)

        print(f"============ End of experiment in condition {mode} ========================")
        return inspect_res, minor_repair_res, major_repair_res, replace_res

    def time_based_benchmark(self, num_iter, mode):
        """
        Time-based benchmark. Actions are performed after predefined intervals. Here, the best
        interval combinations are calculated.
        :param num_iter: int - the iteration number
        :param mode: str - the type of benchmark (f.i. time, condition)
        :return: all_costs: ndarray - numpy array with all the costs of the experiment
        """
        self._load_params(self.param_dict[mode])

        if not self.repair_interval_min:
            self.repair_interval_min = -1
        if not self.replace_interval_min:
            self.replace_interval_min = -1
        if not self.inspect_interval_min:
            self.inspect_interval_min = -1

        if not self.repair_interval_max:
            self.repair_interval_max = self.env.timesteps + 2
        if not self.replace_interval_max:
            self.replace_interval_max = self.env.timesteps + 2
        if not self.inspect_interval_max:
            self.inspect_interval_max = self.env.timesteps + 2

        repair_intervals = np.arange(np.max([1, self.repair_interval_min]),
                                     np.min([self.env.timesteps + 1, self.repair_interval_max]))
        replace_intervals = np.arange(np.max([1, self.replace_interval_min]),
                                      np.min([self.env.timesteps + 1, self.replace_interval_max]))
        inspect_intervals = np.arange(np.max([1, self.inspect_interval_min]),
                                      np.min([self.env.timesteps + 1, self.inspect_interval_max]))

        all_costs = np.zeros((len(inspect_intervals) + 1,
                              len(repair_intervals) + 1,
                              len(replace_intervals) + 1))

        for inspect_idx, inspect_int in tqdm(enumerate(inspect_intervals)):
            for repair_idx, repair_int in enumerate(repair_intervals):
                for replace_idx, replace_int in enumerate(replace_intervals):
                    episode_cost = self.time_based_rule(inspect_int, repair_int, replace_int)

                    all_costs[inspect_idx+1, repair_idx+1, replace_idx+1] = episode_cost

        _, _, _ = self.calculate_min(all_costs, num_iter, mode)

        return all_costs

    def time_based_rule(self, inspect_int, repair_int, replace_int):
        """
        Time based rule
        """
        self.env.reset()
        episode_cost = 0

        for timestep in range(1, self.env.timesteps + 1):
            action = 0
            if timestep % replace_int == 0:
                action = 2
            elif timestep % repair_int == 0:
                action = 1
            if timestep % inspect_int == 0:
                action += 3
            _, step_cost, _, _ = self.env.step([action] * self.env.num_components)

            episode_cost += step_cost * self.env.gamma**timestep

        return episode_cost

    def condition_based_benchmark(self, num_iter, mode):
        """
        Condition based benchmark. Action are performed when the condition of the component has
        reached a specific threshold. Here, the thresholds are calculated.
        :param num_iter: int - the iteration number
        :param mode: str - the type of benchmark (f.i. time, condition)
        :return: all_costs: ndarray - numpy array with all the costs of the experiment
        """
        self._load_params(self.param_dict[mode])
        self.env.num_states = self.env.num_states_iri

        if not self.minor_repair_threshold_min:
            self.minor_repair_threshold_min = -1
        if not self.major_repair_threshold_min:
            self.major_repair_threshold_min = -1
        if not self.replace_threshold_min:
            self.replace_threshold_min = -1
        if not self.inspect_interval_min:
            self.inspect_interval_min = -1

        if not self.minor_repair_threshold_max:
            self.minor_repair_threshold_max = self.env.num_states + 1
        if not self.major_repair_threshold_max:
            self.major_repair_threshold_max = self.env.num_states + 1
        if not self.replace_threshold_max:
            self.replace_threshold_max = self.env.num_states + 1
        if not self.inspect_interval_max:
            self.inspect_interval_max = self.env.timesteps + 2

        minor_repair_thresholds = np.arange(np.max([0, self.minor_repair_threshold_min]),
                                      np.min([self.env.num_states + 1,
                                              self.minor_repair_threshold_max]))
        major_repair_thresholds = np.arange(np.max([0, self.major_repair_threshold_min]),
                                            np.min([self.env.num_states + 1,
                                                    self.major_repair_threshold_max]))
        replace_thresholds = np.arange(np.max([0, self.replace_threshold_min]),
                                       np.min(
                                           [self.env.num_states + 1, self.replace_threshold_max]))
        inspect_intervals = np.arange(np.max([1, self.inspect_interval_min]),
                                      np.min([self.env.timesteps + 1, self.inspect_interval_max]))

        all_costs = np.zeros((len(inspect_intervals) + 1,
                              len(minor_repair_thresholds),
                              len(major_repair_thresholds),
                              len(replace_thresholds)))

        for inspect_int in tqdm(inspect_intervals):
            for minor_repair_thres in minor_repair_thresholds:
                for major_repair_thres in major_repair_thresholds:
                    for replace_thres in replace_thresholds:

                        episode_cost, _ = self.condition_based_rule(inspect_int, minor_repair_thres,
                                                                 major_repair_thres, replace_thres)
                        all_costs[inspect_int, minor_repair_thres, major_repair_thres,
                        replace_thres] = episode_cost

        _, _, _, _ = self.calculate_min(all_costs, num_iter, mode)

        return all_costs

    def condition_based_rule(self, inspect_int, minor_repair_thres, major_repair_thres,
        replace_thres):
        self.env.reset()
        episode_cost = np.zeros(3)

        for timestep in range(1, self.env.timesteps + 1):
            action = np.zeros(self.env.num_components, dtype=int)
            for comp in range(self.env.num_components):
                # mean_state_cci = np.random.choice(range(self.env.states_cci.shape[1]), 1,
                #                               p=self.env.states_cci[comp])
                mean_state_iri = np.random.choice(range(self.env.states_iri.shape[1]), 1,
                                                  p=self.env.states_iri[comp])
                mean_state = np.max([mean_state_iri])
                if mean_state >= replace_thres:
                    action[comp] = 4
                elif mean_state >= major_repair_thres:
                    action[comp] = 1
                # elif mean_state >= minor_repair_thres:
                #     action[comp] = 1
                if (timestep % inspect_int == 0) and (action[comp] != 4):
                    action[comp] += 2

            _, step_cost, _, _ = self.env.step(self.env.actions[action])

            episode_cost += step_cost * self.env.gamma**(timestep-1)

        return np.dot(episode_cost, self.env.w_rewards), episode_cost

    @staticmethod
    def calculate_min(costs, num_iter=0, mode=""):
        """
        Calculate the best combination of thresholds/intervals from all the experiments
        :param costs: ndarray - all costs from all experiments
        :param num_iter: int - iteration number, in case of one input cost
        :param mode: str - the type of benchmark (f.i. time, condition)
        :return: -
        """
        min_cost = np.max(costs[np.nonzero(costs)])
        min_intervals = np.where(costs == min_cost)
        # print(f"Mode {mode} - Iteration: {num_iter}")
        print(f"Min cost = {min_cost}")
        print(f"Inspection : Every {min_intervals[0][0]} years")
        # print(f"Minor Repair : When IRI hits {min_intervals[1][0]}")
        print(f"Major Repair : When IRI hits {min_intervals[2][0]}")
        print(f"Replace = When IRI hits {min_intervals[3][0]}")

        return min_intervals[0][0], min_intervals[1][0], min_intervals[2][0], min_intervals[3][0]

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


if __name__ == "__main__":
    os.chdir("../")
    benchmarks = Benchmarks()
    benchmarks.execute_benchmarks()
    #
    # ppo = MindmapPPO()
    # ppo.run_episodes(exec_mode="train")
    os.chdir("../")
    # ppo = MindmapPPO()
    # ppo.run_episodes(exec_mode="train")
    # ppo.run_episodes(exec_mode="test", checkpoint_dir="src/model_weights/20230213155645/",
    #                  checkpoint_ep=1000)
