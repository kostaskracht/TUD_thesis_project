import numpy as np
import os
import thesis_env
import gym
from datetime import datetime
import time
from tqdm import tqdm
import yaml

from parallel_execution_new import MindmapPPOMultithread

os.chdir("../")
env = gym.make("thesis-env-v1", quiet=True)

import logging

logging.disable(logging.WARNING)
# os.chdir("../../../..")

episodes = 100
results = []

ppo = MindmapPPOMultithread()
# maintenance cost
# checkpoint_dir = "../results/greenlight/ols_no_reuse/outputs/seed1244/20230401165818_954/ppo/20230401165818_088/model_weights/"
# checkpoint_ep = 13600

# # carbon
# checkpoint_dir = "../results/greenlight/ols_no_reuse/outputs/seed1244/20230401165818_954/ppo/20230401174326_940/model_weights/"
# checkpoint_ep = 14600
#
# # user cost
# checkpoint_dir = "../results/greenlight/ols_no_reuse/outputs/seed1244/20230401165818_954/ppo/20230401183011_940/model_weights/"
# checkpoint_ep = 13800

# # ~0.33 all
checkpoint_dir = "../results/greenlight/ols_no_reuse/outputs/seed1234/20230401230311_451/ppo/20230402051630_815/model_weights/"
checkpoint_ep = 14999

ppo._load_model_weights(checkpoint_dir=checkpoint_dir,
                        checkpoint_ep=checkpoint_ep, reuse_mode="full")

mode = "random" # ppo, cbm, random, other

# cbm:
# cost: inspect int: 1, minor_repair res: 0, major_repair res: 2,replace res: 4
# carbon:inspect int: 1, minor_repair res: 0, major_repair res: 2,replace res: 5
# user: inspect int: 8, minor_repair res: 0, major_repair res: 3,replace res: 5

rewards_basket = []
all_actions = []
all_costs = []
all_states = []
all_cost_comps = {}
all_carbon_comps = {}
all_user_comps = {}

for ep in range(episodes):

    # Sample policy check

    # Initialize tables for visualization
    actions = []
    costs = []
    states_iri = []
    traffic = []
    episode_cost = np.zeros(3)

    cost_comps = []
    carbon_comps = []
    user_comps = []

    # total_urgent_comps = 0

    tau = 0
    import time

    env.reset()
    env.num_traffic_assignments = 0
    begin_time = time.time()
    for i in range(1, 21):
        # for i in range(1, 4):

        step_time = time.time()

        # CBM specific
        inspect_interval = 8
        repair_state = 3
        replace_state = 5

        if mode == "random":
            # # Random actions
            cur_action = np.random.choice(np.arange(5), size=env.num_components, replace=True,
                                          # p=[0.8] + [0.2/(env.num_actions-1)]*(env.num_actions-1)
                                          )
        elif mode == "ppo":
            # Load action from PPO output
            cur_action, _ = ppo.actor.sample_action_actor(env.states_nn, ep=1)

        elif mode == "cbm":
            # CBM best setting
            cur_action = np.zeros(env.num_components, dtype=int)
            for comp in range(env.num_components):
                mean_state_iri = np.random.choice(range(env.states_iri.shape[1]), 1,
                                                  p=env.states_iri[comp])
                mean_state = np.max([mean_state_iri])
                if mean_state >= replace_state:
                    cur_action[comp] = 4
                elif mean_state >= repair_state:
                    cur_action[comp] = 1
                # elif mean_state >= minor_repair_thres:
                #     action[comp] = 1
                if (i % inspect_interval == 0) and (cur_action[comp] != 4):
                    cur_action[comp] += 2

        else:
            pass

        cur_action = np.array([0]*10)
        states, step_cost, done, metadata = env.step(env.actions[cur_action])
        rewards_basket.append(step_cost)

        # actions.append(cur_action)
        episode_cost += (env.gamma ** (i-1)) * step_cost[0] * env.norm_factor[0]
        # total_urgent_comps += len(env.urgent_comps)
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
        from copy import copy
        actions.append(copy(metadata["actions"]))
        costs.append(copy(metadata["costs"]))
        states_iri.append(copy(metadata["states_iri"]))
        cost_comps.append(copy(metadata["cost_components"]))
        carbon_comps.append(copy(metadata["carbon_components"]))
        user_comps.append(copy(metadata["convenience_components"]))

        # # states_cci.append(metadata["states_cci"])
        # states_iri.append(metadata["states_iri"])
        # traffic.append(metadata["traffic"])
    results.append([ep, time.time() - begin_time, env.num_traffic_assignments])
    # print(f"Episode {ep}: {time.time() - begin_time}, {env.num_traffic_assignments}")

    all_costs.append(np.array(costs).flatten())
    all_actions.append(np.array(actions).flatten())
    all_cost_comps[ep] = np.array(cost_comps).flatten().tolist()
    all_carbon_comps[ep] = np.array(carbon_comps).flatten().tolist()
    all_user_comps[ep] = np.array(user_comps).flatten().tolist()
    all_states.append(np.array(states_iri).flatten().tolist())

    act, counts = np.unique(np.stack(actions), return_counts=True)
    print(f"Episode {ep}: "
          f"Total reward is: {episode_cost[0]}"
          f" Actions percentages {dict(zip(act.astype(int), counts*100//(env.num_components*env.timesteps)))}"
          # f"Total urgent comps {total_urgent_comps}"
          )

print(f"Average cost is {np.mean(all_costs)}")

rew_bas = np.stack(rewards_basket)
means = np.mean(rew_bas, axis=0)
std = np.std(rew_bas, axis=0)
minn = np.min(rew_bas, axis=0)
maxx = np.max(rew_bas, axis=0)

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
# np.savetxt("timing_thesis.csv", results, delimiter=",")
# np.savetxt("greenlight_even_rew.csv", all_costs, delimiter=",")
# np.savetxt("greenlight_even_action.csv", all_actions, delimiter=",")
# np.savetxt("greenlight_even_states.csv", all_states, delimiter=",")
import json
json.dump(all_cost_comps, open("greenlight_ppo_even_cost_comp.json", "w"))
json.dump(all_carbon_comps, open("greenlight_ppo_even_carbon_comp.json", "w"))
json.dump(all_user_comps, open("greenlight_ppo_even_user_comp.json", "w"))
