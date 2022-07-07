import numpy as np
import agent
import environment

episodes = 3000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [9, 7]           # objective point
discount = 0.9          # exponential discount factor
softmax = True         # set to true to use Softmax policy
sarsa = True           # set to true to use the Sarsa algorithm

#%% grid search for parameters tuning

# Parameters grid
alpha_1 = np.linspace(0.99,0.01,episodes)
alpha_2 = np.linspace(0.09,0.00001,episodes)
alpha_3 = np.ones(episodes) * 0.05
epsilon_1 = np.linspace(0.99, 0.001, episodes)
epsilon_2 = np.linspace(0.8, 0.001, episodes)
epsilon_3 = np.linspace(0.7, 0.001, episodes)

alpha_vec = [alpha_1, alpha_2, alpha_3]
epsilon_vec = [epsilon_1, epsilon_2, epsilon_3]

star_alpha = 0
star_epsilon = 0

num_star_alpha = 0
num_star_epsilon = 0

#obstacles 
obstacle_y = [2,3,4,5,6,7,8]
obstacle_x = [4]
obstacle_2 = [6,7,8]


#%% Q-learning
softmax = False
sarsa = False

# perform the grid search for the Q-learning
star_reward = -1000
reward_4_net = []
num_star_alpha = 0
num_alpha = 0
# only alpha change, epsilon in fixed
epsilon = epsilon_1

for alpha in alpha_vec:
    learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
    num_alpha +=1
    reward_4_episode = []
    for index in range(0, episodes):
        # start from a random state
        while(True):
            initial = [np.random.randint(0, 3), np.random.randint(3, y)]
            if(not(initial[0] in obstacle_x and initial[1] in obstacle_y 
                   and initial[0] in obstacle_2 and initial[1] in obstacle_2)):
                break
            
        state = initial
        env = environment.Environment(x, y, state, goal)
        reward = 0
        # run episode
        for step in range(0, episode_length):
            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = learner.select_action(state_index, epsilon[index])
            # the agent moves in the environment
            result = env.move(action)
            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
            # update state and reward
            reward += result[1]
            state = result[0]
        reward /= episode_length
        reward_4_episode.append(reward)
    mean_reward = np.mean(reward_4_episode)
    print('\n mean_reward ' +str(mean_reward))
    if(mean_reward > star_reward):
        star_reward = mean_reward
        star_alpha = alpha
        num_star_alpha = num_alpha

print('\t q-learnign best alpha is: ' + str(star_alpha[0]) + ' ... ' + str(star_alpha[-1]))
print('\t q-learnign best alpha is number: ' + str(num_star_alpha))


#%% Sarsa with greedy
softmax = False
sarsa = True

# perform the grid search for the Sarsa with greedy
star_reward = -1000
reward_4_net = []
num_star_alpha = 0
num_alpha = 0
num_star_epsilon = 0
num_epsilon = 0

for epsilon in epsilon_vec:
    num_epsilon +=1
    num_alpha = 0
    for alpha in alpha_vec:
        learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
        num_alpha +=1
        reward_4_episode = []
        for index in range(0, episodes):
            # start from a random state
            while(True):
                initial = [np.random.randint(0, x), np.random.randint(0, y)]
                if(not(initial[0] in obstacle_x and initial[1] in obstacle_y 
                       and initial[0] in obstacle_2 and initial[1] in obstacle_2)):
                    break
                
            state = initial
            env = environment.Environment(x, y, state, goal)
            reward = 0
            # run episode
            for step in range(0, episode_length):
                # find state index
                state_index = state[0] * y + state[1]
                # choose an action
                action = learner.select_action(state_index, epsilon[index])
                # the agent moves in the environment
                result = env.move(action)
                # Q-learning update
                next_index = result[0][0] * y + result[0][1]
                learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                # update state and reward
                reward += result[1]
                state = result[0]
            reward /= episode_length
            reward_4_episode.append(reward)
        mean_reward = np.mean(reward_4_episode)
        print('\n mean_reward ' +str(mean_reward))
        if(mean_reward > star_reward):
            star_reward = mean_reward
            star_alpha = alpha
            star_epsilon = epsilon
            num_star_alpha = num_alpha
            num_star_epsilon = num_epsilon
            
print('\t Sarsa with greedy best alpha is: ' + str(star_alpha[0]) + ' ... ' + str(star_alpha[-1]))
print('\t Sarsa with greedy best alpha is number: ' + str(num_star_alpha))
print('\t Sarsa with greedy best epsilon is: ' + str(star_epsilon[0]) + ' ... ' + str(star_epsilon[-1]))
print('\t Sarsa with greedy best epsilon is number: ' + str(num_star_epsilon))

#%% Sarsa with softmax
softmax = True
sarsa = True

# perform the grid search for the Sarsa with greedy
star_reward = -1000
reward_4_net = []
num_star_alpha = 0
num_alpha = 0
num_star_epsilon = 0
num_epsilon = 0

for epsilon in epsilon_vec:
    num_epsilon +=1
    num_alpha = 0
    for alpha in alpha_vec:
        learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
        num_alpha +=1
        reward_4_episode = []
        for index in range(0, episodes):
            # start from a random state
            while(True):
                initial = [np.random.randint(0, x), np.random.randint(0, y)]
                if(not(initial[0] in obstacle_x and initial[1] in obstacle_y 
                       and initial[0] in obstacle_2 and initial[1] in obstacle_2)):
                    break
                
            state = initial
            env = environment.Environment(x, y, state, goal)
            reward = 0
            # run episode
            for step in range(0, episode_length):
                # find state index
                state_index = state[0] * y + state[1]
                # choose an action
                action = learner.select_action(state_index, epsilon[index])
                # the agent moves in the environment
                result = env.move(action)
                # Q-learning update
                next_index = result[0][0] * y + result[0][1]
                learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                # update state and reward
                reward += result[1]
                state = result[0]
            reward /= episode_length
            reward_4_episode.append(reward)
        mean_reward = np.mean(reward_4_episode)
        print('\n mean_reward ' +str(mean_reward))
        if(mean_reward > star_reward):
            star_reward = mean_reward
            star_alpha = alpha
            star_epsilon = epsilon
            num_star_alpha = num_alpha
            num_star_epsilon = num_epsilon
            
print('\t Sarsa with softmax best alpha is: ' + str(star_alpha[0]) + ' ... ' + str(star_alpha[-1]))
print('\t Sarsa with softmax best alpha is number: ' + str(num_star_alpha))
print('\t Sarsa with softmax best epsilon is: ' + str(star_epsilon[0]) + ' ... ' + str(star_epsilon[-1]))
print('\t Sarsa with softmax best epsilon is number: ' + str(num_star_epsilon))

















