# OLS algorithm: Derived from
import io
from itertools import combinations
from typing import List, Optional
from datetime import datetime
import json
from numpyencoder import NumpyEncoder
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import yaml

# import wandb as wb
from utils import random_weights, hypervolume, policy_evaluation_mo, sparsity

# from q_learning_cartpole import Qtable, Q_learning
import gym
# import cartpole_envi
import torch as th

import sys
# sys.path.append('/')
from parallel_execution_new import MindmapPPOMultithread
from Benchmarks import Benchmarks

import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

np.set_printoptions(precision=4)

import logging

class OLS:
    # Section 3.3 of http://roijers.info/pub/thesis.pdf
    def __init__(
            self,
            m: int,
            epsilon: float = 0.0,
            max_value: Optional[float] = None,
            min_value: Optional[float] = None,
            reverse_extremum: bool = False,
    ):
        self.m = m # number of objectives
        self.epsilon = epsilon # TODO
        self.W = [] # list of visited weights
        self.ccs = [] # S: partial CCS
        self.ccs_weights = [] # list of the weights from the partial CCS
        self.queue = [] # Priority queue
        self.iteration = 0 # Iteration num
        self.max_value = max_value # TODO
        self.min_value = min_value # TODO
        self.worst_case_weight_repeated = False # TODO
        extremum_weights = reversed(self.extrema_weights()) if reverse_extremum else self.extrema_weights()
        for w in extremum_weights: # For each extrema add the extrema with infinite priority as a
            # tuple (priority, weight)
            self.queue.append((float("inf"), w))

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(np.random.choice(1000)).zfill(3)
        self.output_dir = f"outputs/ols/{self.timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(self.output_dir + "/logs")
            os.makedirs(self.output_dir + "/figs")
            os.makedirs(self.output_dir + "/iters")

    # Select the next weight to process. It will be the first weight in the queue
    def next_w(self) -> np.ndarray:
        return self.queue.pop(0)[1]

    def get_ccs_weights(self) -> List[np.ndarray]:
        return self.ccs_weights.copy()

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        weights = [w for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    # Check whether te priority Q is empty or TODO
    def ended(self) -> bool:
        return len(self.queue) == 0 or self.worst_case_weight_repeated

    # Add processed weight to the list of visited weights W.
    # If the newly found value "value" is dominated by any of the values in "self.CCS", don't add it
    # to the Coverage Set
    # If the newly found value "value" is very very close to another value in "self.CCS",
    # don't add it to the Coverage Set
    # Remove the obsolete weights (the ones that are not corner weights any more)
    # Remove the values that are dominated by the new value in all visited weights
    # Compute the new corner weights and add them to the queue
    # Compute the priority of the weights in the queue
    # Return the indices of the CCS values that were deleted
    def add_solution(self, value, w, gpi_agent=None, env=None) -> int:
        self.iteration += 1
        # Add weight to visited weights W
        self.W.append(w)
        # Check if value is dominated by another one
        if self.is_dominated(value):
            return [len(self.ccs)]
        # Check if value is very very close to another one
        for i, v in enumerate(self.ccs):
            if np.allclose(v, value):
                return [len(self.ccs)]  # delete new policy as it has same value as an old one

        # Get the weights that are obsolated i.e. the ones that are not corner weights any more
        W_del = self.remove_obsolete_weights(new_value=value)
        W_del.append(w)
        logging.info(f"W_del {W_del}")

        # Remove the values from CSS that are dominated by the new value in all visited weights.
        # Return the indices of those values in CSS
        removed_indx = self.remove_obsolete_values(value)

        # Get the new corner weights between the new value and the other CCS values
        W_corner = self.new_corner_weights(value, W_del)

        # Append the new value to the CCS values together with its respective weights
        self.ccs.append(value)
        self.ccs_weights.append(w)

        # Compute the priority (predicted improvement) of each of the new corners, and sort the
        # queue base on descending priority
        # logging.info(f"W_corner {W_corner}")
        for wc in W_corner:
            # gpi_agent is None!
            priority = self.get_priority(wc, gpi_agent, env)
            logging.info(f"W_corner {wc} : Improvement {priority}")
            if priority > self.epsilon:
                self.queue.append((priority, wc))
        # Sort the queue
        self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority

        logging.info(f"queue size {len(self.queue)}")
        logging.info(f"ccs: {self.ccs}")
        logging.info(f"ccs size: {len(self.ccs)}")

        return removed_indx

    # Given a new weight w, compute its estimated improvement (priority)
    def get_priority(self, w, gpi_agent=None, env=None) -> float:
        # Find the maximum potential value with  weight w
        max_optimistic_value = self.max_value_lp(w)
        # Get the maximym scalarized value from the partial CSS values, for the weight w.
        max_value_ccs = self.max_scalarized_value(w)
        # upper_bound_nemecek = self.upper_bound_policy_caches(w)
        # logging.info(f'optimistic: {max_optimistic_value} policy_cache_up: {upper_bound_nemecek}')
        if gpi_agent is not None:
            gpi_value = policy_evaluation_mo(gpi_agent, env, w, rep=1)
            gpi_value = np.dot(gpi_value, w)
            logging.info(f"optimistic: {max_optimistic_value:.4f} smp: {max_value_ccs:.4f} gpi: {gpi_value:.4f}")
            # max_value_ccs = max(max_value_ccs, gpi_value)
        # The priority is the difference between the potential and the current maximum value in
        # the weight.
        priority = max_optimistic_value - max_value_ccs  # / abs(max_optimistic_value)
        return priority

    # Get the maximum scalarized value from the partial CSS values, given a certain weight -->
    # V*s(w).
    def max_scalarized_value(self, w: np.ndarray) -> float:
        if not self.ccs:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def get_set_max_policy_index(self, w: np.ndarray) -> int:
        if not self.ccs:
            return None
        return np.argmax([np.dot(v, w) for v in self.ccs])

    # This function removes the obsolete weights from the queue list. An obsolete weight is a
    # weight that is not in the corner anymore! Returns a list with the weights that are obsolated
    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []
        W_del = []
        inds_remove = []
        # Iterate over the (priority, weight) sets inside the queue list
        for i, (priority, cw) in enumerate(self.queue):
            # Check if sum(i from 0 to m-1) (cw[i] * new_value[i]) is larger than V*s(w). If
            # so, this means that cw is not a corner point any more!
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):  # and priority != float('inf'):
                W_del.append(cw)
                inds_remove.append(i)
        # Remove the obsolete weights from the queue
        for i in reversed(inds_remove):
            self.queue.pop(i)
        return W_del

    # Remove the values from CSS that have a smaller scalarization than the new value for all
    # visited weights.
    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        removed_indx = []
        # Iterate over the values
        for i in reversed(range(len(self.ccs))):
            best_in_all = True
            # Iterate over the visited weights W, and check whether the new value is superior in
            # all visited weights
            for j in range(len(self.W)):
                w = self.W[j]
                # Check if V*new_value(w) < V*cur_value(w)
                if np.dot(value, w) < np.dot(self.ccs[i], w):
                    best_in_all = False
                    break
            if best_in_all:
                # If the current value is dominated by the new value in all weights, remove it!
                logging.info(f"Removed value {self.ccs[i]}")
                removed_indx.append(i)
                self.ccs.pop(i)
                self.ccs_weights.pop(i)
        return removed_indx

    # Find the maximum possible value vector for w_new, so that this new value doesn't dominate
    # any other weights
    def max_value_lp(self, w_new: np.ndarray) -> float:
        # Check if the CCS is empty
        if len(self.ccs) == 0:
            return float("inf")

        w = cp.Parameter(self.m)
        w.value = w_new
        v = cp.Variable(self.m)
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Maximize(w @ v)
        constraints = [W @ v <= V]
        if self.max_value is not None:
            constraints.append(v <= self.max_value)
        if self.min_value is not None:
            constraints.append(v >= self.min_value)
        prob = cp.Problem(objective, constraints)
        return prob.solve(verbose=False)

    def upper_bound_policy_caches(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        alpha = cp.Variable(len(self.W))
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Minimize(alpha @ V)
        constraints = [alpha @ W == w, alpha >= 0]
        prob = cp.Problem(objective, constraints)
        upper_bound = prob.solve()
        if prob.status == cp.OPTIMAL:
            return upper_bound
        else:
            return float("inf")

    # Function not used anywhere in the code
    def worst_case_weight(self) -> np.ndarray:
        if len(self.W) == 0:
            return random_weights(dim=self.m)
        w = None
        min = float("inf")
        w_var = cp.Variable(self.m)
        params = []
        for v in self.ccs:
            p = cp.Parameter(self.m)
            p.value = v
            params.append(v)
        for i in range(len(self.ccs)):
            objective = cp.Minimize(w_var @ params[i])
            constraints = [0 <= w_var, cp.sum(w_var) == 1]
            # constraints = [cp.norm(w_var) - 1 <= 0, 0 <= w_var]
            for j in range(len(self.ccs)):
                if i != j:
                    constraints.append(w_var @ (params[j] - params[i]) <= 0)
            prob = cp.Problem(objective, constraints)
            value = prob.solve()
            if value < min and prob.status == cp.OPTIMAL:
                min = value
                w = w_var.value.copy()

        if np.allclose(w, self.W[-1]):
            self.worst_case_weight_repeated = True

        return w

    # Given the new value and the weights to be deleted (obsolate weights + most recently used
    # weight) this function computes the new corner weights between the new value and the rest of
    # the values.
    def new_corner_weights(self, v_new: np.ndarray, W_del: List[np.ndarray]) -> List[np.ndarray]:
        # If no values have been stored in CCS yet, return
        if len(self.ccs) == 0:
            return []
        V_rel = []
        W_new = []
        # Iterate over the deleted weights
        for w in W_del:
            best = [self.ccs[0]]
            # Iterate over the values of CCS to find the values for which this weight is maximized
            for v in self.ccs[1:]:
                if np.allclose(np.dot(w, v), np.dot(w, best[0])):
                    best.append(v)
                elif np.dot(w, v) > np.dot(w, best[0]):
                    best = [v]
            # Append the best values for the current weight
            V_rel += best
            if len(best) < self.m:
                # Find a new corner weight for the current deleted one
                wc = self.corner_weight(v_new, best)
                # Add the new weight to the W_new list
                W_new.append(wc)
                # Add the exrema weights again ([1, 0, 0], [0, 1, 0], [0, 0, 1])
                W_new.extend(self.extrema_weights())

        # Now, V-rel contains all value vectors from CSS that had their corner weights affected
        V_rel = np.unique(V_rel, axis=0)
        # V_rel = self.ccs.copy()
        # Find the corner weights of all combinations of affected weights with the new value
        for comb in range(1, self.m):
            for x in combinations(V_rel, comb):
                if not x:
                    continue
                wc = self.corner_weight(v_new, x)
                W_new.append(wc)

        # From the newly added weights, only keep the ones that are not:
        # - "None",
        # - Identical to an already visited point inside W
        # - In the queue to be visited
        filter_fn = lambda wc: (wc is not None) and (not any([np.allclose(wc, w_old) for w_old in self.W] + [np.allclose(wc, w_old) for p, w_old in self.queue]))
        # (np.isclose(np.dot(wc, v_new), self.max_scalarized_value(wc))) and \
        W_new = list(filter(filter_fn, W_new))
        W_new = np.unique(W_new, axis=0)
        return W_new

    # Find the new corner weight between the new value and the rest of the values, for all corner
    # weights that have been deleted.
    # Solve a linear problem to find the new weight (wc), constraining the scalarized values of
    # the new value and the selected values to be equal!
    def corner_weight(self, v_new: np.ndarray, v_set: List[np.ndarray]) -> np.ndarray:
        wc = cp.Variable(self.m)
        v_n = cp.Parameter(self.m)
        v_n.value = v_new
        objective = cp.Minimize(v_n @ wc)  # cp.Minimize(0)
        constraints = [0 <= wc, cp.sum(wc) == 1]
        for v in v_set:
            v_par = cp.Parameter(self.m)
            v_par.value = v
            constraints.append(v_par @ wc == v_n @ wc)
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)  # (solver='SCS', verbose=False, eps=1e-5)
        if prob.status == cp.OPTIMAL:
            weight = np.clip(wc.value, 0, 1)  # ensure range [0,1]
            weight /= weight.sum()  # ensure sum to one
            return weight
        else:
            return None

    # CHECKED
    def extrema_weights(self) -> List[np.ndarray]:
        # appends the extreme weights (e.g. [1, 0, 0], [0, 1, 0], [0, 0, 1] for m=3)
        # returns a list of np.arrays --> [array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])]
        extrema_weights = []
        for i in range(self.m):
            w = np.zeros(self.m)
            w[i] = 1.0
            extrema_weights.append(w)
        return extrema_weights

    # Check if the newly found value is pareto dominated by any of the rest of the values:
    # If the newly found value is smaller than another value in all objectives, return True.
    # Else, return False
    def is_dominated(self, value):
        for v in self.ccs:
            if (v > value).all():
                return True
        return False

    def plot_ccs(self, ccs, ccs_weights, writer=None, gpi_agent=None, eval_env=None):
        import seaborn as sns
        params = {
            "text.latex.preamble": r"\usepackage{amsmath}",
            "mathtext.fontset": "cm",
            "figure.figsize": (1.5 * 4.5, 1.5 * 3),
            "xtick.major.pad": 0,
            "ytick.major.pad": 0,
        }
        plt.rcParams.update(params)
        sns.set_style("white", rc={"xtick.bottom": False, "ytick.left": False})
        sns.set_context(
            "paper",
            rc={
                "text.usetex": True,
                "lines.linewidth": 2,
                "font.size": 15,
                "figure.autolayout": True,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "axes.titlesize": 12,
                "axes.labelsize": 15,
                "lines.markersize": 12,
                "legend.fontsize": 8,
            },
        )
        colors = ["#E6194B", "#5CB5FF"]
        sns.set_palette(colors)

        # x_css, y_css = [], []
        # for i in range(len(ccs)):
        #     x_css.append(ccs[i][0])
        #     y_css.append(ccs[i][1])

        if len(self.ccs[0]) > 2:
            flag_3d = True
        else:
            flag_3d = False

        # x, y = [], []
        # for i in range(len(self.ccs)):
        #     x.append(self.ccs[i][0])
        #     y.append(self.ccs[i][1])

        ccs_arr = np.stack(self.ccs)
        x = ccs_arr[:, 0]
        y = ccs_arr[:, 1]

        fig = plt.figure()
        if flag_3d:
            z = ccs_arr[:, 2]
            ax = fig.add_subplot(projection='3d')
            ax.scatter(
                x,
                y,
                z,
                label="$\Psi$ (SF set at iteration {})".format(self.iteration),
                marker="^",
                color=colors[1],
            )
        else:
            ax = fig.add_subplot()
            ax.scatter(
                x,
                y,
                label="$\Psi$ (SF set at iteration {})".format(self.iteration),
                marker="^",
                color=colors[1],
            )

        # plt.ylim(max(y), min(y))
        # plt.scatter(x_css, y_css, label="CCS", marker="x", color="black")
        ax.legend(loc="best", fancybox=True, framealpha=0.5)
        ax.set_xlabel("$\psi^{\pi}_{1}$ (Lifecycle cost)")
        ax.set_ylabel("$\psi^{\pi}_{2}$ (Lifecycle carbon emissions)")

        if flag_3d:
            ax.set_zlabel("$\psi^{\pi}_{3}$ (User cost)")

        sns.despine()
        plt.grid(alpha=0.25)
        # plt.tight_layout()
        # plt.savefig(f"figs/ccs_dst{self.iteration}.pdf", format="pdf", bbox_inches="tight")

        if writer:
            # Log to tensorboard
            fig.canvas.draw()

            # Convert the figure to numpy array, read the pixel values and reshape the array
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img_arr = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            writer.add_image("CCS", img_arr, dataformats="HWC", global_step=self.iteration)

        plt.savefig(f"{self.output_dir}/figs/ccs_dst{self.iteration}.pdf", format="pdf")
        fig.show()

        # wb.log(
        #     {
        #         "metrics/ccs": wb.Image(plt),
        #         "global_step": gpi_agent.policies[-1].num_timesteps,
        #         "iteration": self.iteration,
        #     }
        # )

    def plot_interactive(self):
        import plotly.express as px
        import pandas as pd

        labels = ["Lifecycle maintenance cost", "Lifecycle carbon emissions", "Lifecycle user cost"]

        ccs_arr = np.stack(self.ccs)
        df = pd.DataFrame({labels[0]: ccs_arr[:, 0], labels[1]: ccs_arr[:, 1], labels[2]: ccs_arr[:, 2]})

        fig = px.scatter_3d(df, x=labels[0], y=labels[1], z=labels[2], opacity=0.7)

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.write_html(f"{self.output_dir}/figs/ccs_dst_interactive.html")
        fig.show()


def find_closest_run(prev_runs_metadata, w):
    min_norm = np.inf
    closest_run = list(prev_runs_metadata.keys())[0]
    for run_timestamp in prev_runs_metadata.keys():
        cur_norm = np.linalg.norm(w - prev_runs_metadata[run_timestamp]["weights"])
        if cur_norm < min_norm:
            min_norm = cur_norm
            closest_run = run_timestamp
    return prev_runs_metadata[closest_run]


def solve(w, ols_out_dir, prev_runs_metadata, reuse_mode, MODEL_FILE=None, ENV_FILE=None):

    ppo_output_dir = f"{ols_out_dir}/ppo"
    ppo = MindmapPPOMultithread(quiet=True, param_file=MODEL_FILE, env_file=ENV_FILE, output_dir=ppo_output_dir)
    benchmarks = Benchmarks()

    enable_ppo = True
    enable_cbm = False
    cap_ppo = False

    # Setting the new preferences
    if len(w) == 2:
        ppo.env.w_rewards = [w[0], w[1], 0.0]  # TODO - only assume 2 objectives
        benchmarks.env.w_rewards = [w[0], w[1], 0.0]
    else:
        ppo.env.w_rewards = w
        benchmarks.env.w_rewards = w
    logging.info(f"Begin execution with weights: {benchmarks.env.w_rewards}")

    if enable_cbm:
        logging.info(f"Saving benchmark execution under {benchmarks.timestamp}")

        logging.info(f"Executing CBM:")
        cbm_value, cbm_values = benchmarks.execute_benchmarks(output_dir=ols_out_dir)
        logging.info(f"CBM return for weights: {benchmarks.env.w_rewards} is {cbm_value}")

        if cap_ppo:
            max_val = cbm_value
        else:
            max_val = None
    else:
        max_val = None

    if enable_ppo:
        logging.info(f"Saving PPO execution under {ppo.timestamp}")
        # if ppo.env.w_rewards[0] == 1.0:
        #     return np.array([-8.5088e+01, -4.2125e+3]), {"best_episode": 400, "output_dir": "src/model_weights/20230304220616_697"}, writer
        # elif ppo.env.w_rewards[1] == 1.0:
        #     return np.array([-1.3583e+03, -0.93151]), {"best_episode": 14200, "output_dir": "src/model_weights/20230305004020_706"}, writer
        if len(prev_runs_metadata) == 0 or reuse_mode == "no":
            ppo.run(exec_mode="train", max_val=max_val)
        else:
            closest_run_metadata = find_closest_run(prev_runs_metadata, w)
            logging.info(f"Closest run is: {closest_run_metadata} with weights {closest_run_metadata['weights']}")
            ppo.run(exec_mode="continue_training", checkpoint=(closest_run_metadata["output_dir"],
                                                               closest_run_metadata["best_episode"]), reuse_mode=reuse_mode,
                    max_val=max_val)

        # Keep the best weight and output dir of current run for the next one
        best_episode = ppo.best_weight
        output_dir = ppo.checkpoint_dir #+ ppo.timestamp + "/"

        logging.info(f"Best return from testing rounds was {ppo.best_result}")

        prev_runs_metadata[ppo.timestamp] = {"output_dir": output_dir, "best_episode": best_episode,
                                             "weights": w}

        logging.info(f"Begin testing:")
        # Get the value of current execution
        iters = ppo.test_episodes
        n_obj = ppo.env.num_objectives
        # values = np.zeros((iters, n_obj))
        ppo.env.reset()
        # for i in range(iters):
        observation = ppo.env.states_nn
        init_observations = th.tensor(np.array(observation), dtype=th.float)
        values = ppo.run(exec_mode="test", checkpoint=(output_dir, best_episode))
        # values[i] = ppo.critic(init_observations).detach().numpy()

        return values[:len(w)], \
            prev_runs_metadata

    else:
        logging.info(f"Returning the CBM value")
        return cbm_values[:len(w)], {}


def execute_OLS(model_file=None, env_file=None, reuse_mode="no", epsilon=0.0001, continue_execution=False,
                file_to_load=None):
    # file_to_load = "outputs/ols/20230327125850_642/iters/iter_1.json"
    m = 3  # number of objectives
    ols = OLS(m=m, epsilon=epsilon)  # , min_value=0.0, max_value=1 / (1 - 0.95) * 1)
    prev_runs_metadata = {}

    # Begging with logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{ols.output_dir}/logs/{ols.timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"OLS execution log file is under {ols.output_dir}/logs/{ols.timestamp}.log")

    logging.info(f"OLS execution started with env file {env_file} and model file {model_file}")
    logging.info(f"OLS execution started with reuse mode {reuse_mode}")
    logging.info(f"OLS execution started with epsilon {epsilon}")

    logging.info(f"OLS execution started with continue execution {continue_execution}")

    # Set up the tensorboard dashboard for OLS execution
    logging.info(f"Began OLS execution {ols.timestamp}")
    writer = SummaryWriter(f"runs/ols/{ols.timestamp}")

    if continue_execution:
        logging.info(f"Loading OLS setting from file {file_to_load}")
        input_dict = json.load(open(file_to_load))
        input_dict["queue"] = [tuple(x) for x in input_dict["queue"]]
        for key, value in input_dict.items():
            if isinstance(value, list) and key != "queue":
                new_val = [np.asarray(xx) for xx in value]
                input_dict[key] = new_val
                # input_dict[key] = np.asarray(value)
        ols.m = input_dict["m"]
        ols.W = input_dict["W"]
        ols.ccs = input_dict["ccs"]
        ols.ccs_weights = input_dict["ccs_weights"]
        ols.queue = input_dict["queue"]
        ols.iteration = input_dict["iteration"]
        # Add the previous runs metadata
        prev_runs_metadata = input_dict["prev_runs_metadata"]

    start = time.time()
    while not ols.ended():
        # Select the weight to process
        w = ols.next_w()
        # logging.info(f"w to check is: {w}")
        # Solve the single objective problem with the given weight
        logging.info(f"==============================================================================")
        logging.info(f"Beginning OLS iteration {ols.iteration}")

        value, prev_runs_metadata = solve(w, ols.output_dir, prev_runs_metadata, reuse_mode, MODEL_FILE=model_file, ENV_FILE=env_file)
        logging.info(f"Value from PPO execution: {value}")
        ols.add_solution(value, w)
        # Plot the convex coverage set
        ols.plot_ccs(ols.ccs, ols.ccs_weights, writer=writer)

        # Save OLS iteration parameters
        output_dict = {"m": ols.m,
                       "W": ols.W,
                       "ccs": ols.ccs,
                       "ccs_weights": ols.ccs_weights,
                       "queue": ols.queue,
                       "iteration": ols.iteration,
                       "prev_runs_metadata": prev_runs_metadata
                       }
        with open(f'{ols.output_dir}/iters/iter_{ols.iteration}.json', "w") as f:
            json.dump(output_dict, f, cls=NumpyEncoder)


        hv = hypervolume(np.zeros(m), ols.ccs)
        sp = sparsity(ols.ccs)
        tm = time.time() - start

        logging.info(f"Hypervolume: {hv:.3f}")
        logging.info(f"Sparsity: {sp:.3f}")
        logging.info(f"Execution time {tm:.2f} seconds.")
        writer.add_scalars("OLS Results",
                           {"Hypervolume": hv,
                            "Sparsity": sp,
                            "Time (sec)": tm},
                           ols.iteration)

    logging.info(f"OLS execution {ols.timestamp} finished.")
    logging.info(f"==============================================================================")

    if len(ols.ccs[0]) > 2:
        ols.plot_interactive()

    return ols

if __name__ == "__main__":
    os.chdir("../")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(np.random.choice(1000)).zfill(3)

    model_file = "src/model_params_mt.yaml"
    env_file = "environments/env_params.yaml"
    reuse_mode = "partial"
    epsilon = 0.0001
    continue_execution = False
    message = ""

    # Update env file parameters
    new_env_params = {
    "seed": 1234
    }

    # Update model file parameters
    new_model_params = {
    "seed": 1234
    }

    files_dict = {env_file: new_env_params, model_file: new_model_params}

    new_filenames = []
    for filename, new_params in files_dict.items():
        # Update the yaml files
        new_filename = filename.replace(".yaml", f"_{timestamp}.yaml")
        with open(filename, 'r') as outfile:
            file_dict = yaml.safe_load(outfile)
            for key, value in new_params.items():
                file_dict[key] = value
        with open(new_filename, 'w+') as outfile:
            yaml.dump(file_dict, outfile, default_flow_style=False)

        new_filenames.append(new_filename)
        print(f"{new_filename}")

    ols = execute_OLS(model_file=new_filenames[1], env_file=new_filenames[0], reuse_mode=reuse_mode, epsilon=epsilon)

    f = open(f"{ols.output_dir}/logs/README", "w")
    f.write(message)
    f.close()
