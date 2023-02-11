import numpy as np
import os
import thesis_env
import gym
from datetime import datetime
from tqdm import tqdm
import yaml

os.chdir("../")
env = gym.make("thesis-env-v1", quiet=True)

import logging

logging.disable(logging.WARNING)
os.chdir("../../../..")

episodes = 10
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
    total_urgent_comps = 0

    tau = 0
    import time

    env.reset()
    env.num_traffic_assignments = 0
    begin_time = time.time()
    for i in range(1, 21):
        # for i in range(1, 4):

        step_time = time.time()
        # cur_action = 0
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
        cur_action = np.zeros(env.num_components, dtype=int)
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

        states, step_cost, done, metadata = env.step(env.actions[cur_action])

        actions.append(cur_action)
        episode_cost += step_cost * env.norm_factor[0]
        total_urgent_comps += len(env.urgent_comps)
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
    # results.append([ep, time.time() - begin_time, env.num_traffic_assignments])
    # print(f"Episode {ep}: {time.time() - begin_time}, {env.num_traffic_assignments}")


    act, counts = np.unique(np.stack(actions), return_counts=True)
    print(f"Total reward is: {episode_cost[0]}"
          f"Actions percentages {dict(zip(act.astype(int), counts*100//(env.num_components*env.timesteps)))}"
          f"Total urgent comps {total_urgent_comps}")
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