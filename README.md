# Reinforcement-Learning
Reinforcement Learning in chess simulation

The aim of this experience was to explore the behaviour of an agent on a 10x10 grid. Through
Reinforcement Learning I analysed the different paths taken by this agent.
The main steps I go through were:
1. creation of the environment, made by the aforementioned grid and 2 obstacle (a rectangular
wall and a square hole) that the agent must learn to avoid while reaching the goal
(file enviroment.py);
2. the creation of the agent’s class, that defines update function and decision policy. Both
Q-learning and Sarsa is implemented. Sarsa has the possibility to use softmax decision
policy instead of greedy, while Q-learning can oly use greedy (epsilon = 0) decision policy
(file agent.py);
3. tuning of alpha and epsilon parameters through a simple grid search (file param-tuning.py);
4. training of the agent and visualization of some test pattern (file training.py).
