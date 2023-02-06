# import libraries
import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
# import pygame
import cartpole_envi

# create environment 
env = gym.make('cartpole-envi-v1')

# Define the Q table
state_space = 4 # number of states
action_space = 2 # number of possible actions

print(env.observation_space.low,"\n",env.observation_space.high)
def Qtable(state_space,action_space,bin_size = 30, num_objectives=None):

    bins = [np.linspace(-4.8,4.8,bin_size),
            np.linspace(-4,4,bin_size),
            np.linspace(-0.418,0.418,bin_size),
            np.linspace(-4,4,bin_size)]

    if num_objectives:
        q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space] + [
            num_objectives]))
    else:
        q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
    return q_table, bins

def Discrete(state, bins):
    index = []
    for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
    return tuple(index)

def Q_learning(q_table, bins, episodes = 5000, gamma = 0.95, lr = 0.1, timestep = 50, epsilon =
0.2, num_objectives=2, weights=[0.5, 0.5]):
    rewards = np.zeros(num_objectives)
    solved = False
    steps = 0
    runs = [0]
    data = {'max' : [0], 'avg' : [0]}
    start = time.time()
    ep = [i for i in range(0,episodes + 1,timestep)]

    for episode in range(1,episodes+1):

        current_state = Discrete(env.reset(),bins) # initial observation
        score = np.zeros(num_objectives)
        done = False
        temp_start = time.time()

        while not done:
            steps += 1
            ep_start = time.time()
            # if episode%episodes == 0:
            #     env.render()

            if np.random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state] @ weights)

            observation, reward, done, info = env.step(action)
            next_state = Discrete(observation,bins)

            score += reward


            if not done:
                max_future_action = np.argmax(q_table[next_state] @ weights)
                max_future_q = q_table[next_state + (max_future_action,)]
                # max_future_q = np.max(q_table[next_state] @ weights)
                #
                current_q = q_table[current_state+(action,)]
                new_q = (1-lr)*current_q + lr*(np.asarray(reward) + gamma*max_future_q)
                q_table[current_state+(action,)] = new_q

            current_state = next_state

        # End of the loop update
        else:
            rewards += score
            runs.append(np.dot(score, weights))
            # if score > 195 and steps >= 100 and solved == False: # considered as a solved:
            if steps >= 195 and solved == False: # considered as a solved:
                solved = True
                print('Solved in episode : {} in time {}'.format(episode, (time.time()-ep_start)))

        # Timestep value update
        if episode%timestep == 0:
            print('Episode : {} | Reward -> {} | Max scalarized reward : {} | Time : {}'.format(
                episode,rewards/timestep, max(runs), time.time() - ep_start))
            data['max'].append(max(runs))
            data['avg'].append(rewards/timestep)
            # if rewards/timestep >= 195:
            #     print('Solved in episode : {}'.format(episode))
            rewards, runs= np.zeros(num_objectives), [0]

    # if len(ep) == len(data['max']):
    #     plt.plot(ep, data['max'], label = 'Max')
    #     plt.plot(ep, data['avg'], label = 'Avg')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Reward')
    #     plt.legend(loc = "upper left")
    #     plt.show()
    init_state = Discrete(env.reset(),bins)
    max_init_action = np.argmax(q_table[init_state] @ weights)
    max_init_q = q_table[init_state + (max_init_action,)]
    return max_init_q
    # env.close()


if __name__ == "__main__":
    q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n, num_objectives=2)

    _ = Q_learning(q_table, bins, lr = 0.15, gamma = 0.995, episodes = 1*10**4, timestep = 100,
               num_objectives=2, weights=[0, 1])
    # env.close()
    # time.sleep(20)
    env.close()