import dill
import numpy as np
import agent
import environment

episodes = 3000        # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [9, 7]           # objective point
discount = 0.9          # exponential discount factor
softmax = False         # set to true to use Softmax policy
sarsa = False           # set to true to use the Sarsa algorithm

# Sarsa with greedy need more time to learn in a proper way the way to the goal
if(sarsa and not(softmax)):
    episodes = 9000

# alpha and epsilon profile - obtained with grid search (param_tuning.py file)
alpha = np.linspace(0.99,0.01,episodes)
epsilon = np.linspace(0.7, 0.001, episodes)
#epsilon = np.ones(episodes) * 0.25

# initialize the agent
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

#obstacles 
obstacle_y = [2,3,4,5,6,7,8]
obstacle_x = [4]
obstacle_2 = [6,7,8]

test_position = [[0,0], [0,9], [0,1], [2,6]];
test_number = 0

moves_record = [];
start_record = []

# perform the training
for index in range(0, episodes):
    # start from a random state
    while(True):
        initial = [np.random.randint(0, 3), np.random.randint(3, y)]
        if(not(initial[0] in obstacle_x and initial[1] in obstacle_y 
               and initial[0] in obstacle_2 and initial[1] in obstacle_2
               )):
            break
    
    # To have a common ground, the final 6 positions are fixed
    if(index > (episodes-1)-len(test_position)):
        initial = test_position[test_number]
        test_number += 1
        
    # initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal)
    reward = 0
    # run episode
    this_moves = [];
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        this_moves.append(result[0]);
        # Q-learning update
        next_index = result[0][0] * y + result[0][1]
        learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # update state and reward
        reward += result[1]
        state = result[0]
    reward /= episode_length
   
    moves_record.append(this_moves)
    start_record.append(initial)
    
    # periodically save the agent
    if ((index + 1) % 10 == 0):
        with open('agent.obj', 'wb') as agent_file:
            dill.dump(agent, agent_file)
        print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial) 

#%%Print of the results
        
for k in range(len(test_position)):
    print('Test position #', k+1)
    treasure_map = moves_record[len(moves_record)-len(test_position)+(k)]
    checkerboard = np.zeros((x,y), dtype=int)
    
    for i in range(len(obstacle_x)):
        for j in range(len(obstacle_y)):
            checkerboard[obstacle_x[i],obstacle_y[j]] = 7
            
    for i in range(len(obstacle_2)):
        for j in range(len(obstacle_2)):
            checkerboard[obstacle_2[i],obstacle_2[j]] = 7
                
    for tmp_position in treasure_map:
        checkerboard[tmp_position[0],tmp_position[1]] += 1
        if (tmp_position[0]==9 and tmp_position[1]==7):
            break
    
    start_tmp = start_record[len(start_record)-len(test_position)+(k)]
    checkerboard[start_tmp[0],start_tmp[1]] = 2
    checkerboard[9,7] = 3
    for i in range(10):
        print(checkerboard[i])
    print("\n\n")
    