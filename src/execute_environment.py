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

episodes = 50
results = []
all_costs = []

ppo = MindmapPPOMultithread()
ppo._load_model_weights(checkpoint_dir="src/model_weights/20230316163006_289/",
                        checkpoint_ep=10250, reuse_mode="full")

mode = "cbm" # ppo, cbm, random, other

for ep in range(episodes):

    # Sample policy check

    all_states = []

    # Initialize tables for visualization
    actions = []
    costs = []
    states_iri = []
    traffic = []
    episode_cost = np.zeros(3)

    # total_urgent_comps = 0

    tau = 0
    import time

    env.reset()
    env.num_traffic_assignments = 0
    begin_time = time.time()
    for i in range(1, 21):
        # for i in range(1, 4):

        step_time = time.time()

        inspect_interval = 1
        repair_state = 2
        replace_state = 4

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

        states, step_cost, done, metadata = env.step(env.actions[cur_action])

        actions.append(cur_action)
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
        # actions.append(metadata["actions"])
        # costs.append(metadata["costs"])
        # # states_cci.append(metadata["states_cci"])
        # states_iri.append(metadata["states_iri"])
        # traffic.append(metadata["traffic"])
    results.append([ep, time.time() - begin_time, env.num_traffic_assignments])
    # print(f"Episode {ep}: {time.time() - begin_time}, {env.num_traffic_assignments}")

    all_costs.append(episode_cost[0])

    act, counts = np.unique(np.stack(actions), return_counts=True)
    print(f"Episode {ep}: "
          f"Total reward is: {episode_cost[0]}"
          f" Actions percentages {dict(zip(act.astype(int), counts*100//(env.num_components*env.timesteps)))}"
          # f"Total urgent comps {total_urgent_comps}"
          )

print(f"Average cost is {np.mean(all_costs)}")

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
np.savetxt("new_timing_comp8_beta3.csv", results, delimiter=",")