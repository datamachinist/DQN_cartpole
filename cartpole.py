"""
Implement DQN for openAI's cart pole environment
Inspired from https://github.com/gsurma/cartpole
"""
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# HYPERPARAMETERS
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
TRAIN_EPISODES = 100
TEST_EPISODES = 10

# INITIALISE ENVIRONMENT
env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
exploration_rate = EXPLORATION_MAX
memory = deque(maxlen=MEMORY_SIZE)

# NEURAL NETWORK
model = Sequential()
model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(action_space, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

def epsilon_policy(state):
    if np.random.rand() < exploration_rate:
        return random.randrange(action_space)   # take a random action
    q_values = model.predict(state)             # else let the NN decide the action
    return np.argmax(q_values[0])

def greedy_policy(state):
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def experience_replay():
    global exploration_rate

    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    for episode, state, action, reward, new_state, done in batch:
        q_update = reward
        if not done:
            q_update = (reward + GAMMA * np.amax(model.predict(new_state)[0]))
        q_values = model.predict(state)
        q_values[0][action] = q_update
        model.fit(state, q_values, verbose=0)

def plot_score(TRAIN_EPISODES, training_rewards):
    x = range(TRAIN_EPISODES)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode number')
    plt.ylabel('Cumulative training reward')
    plt.show()
    
def play_best(training_rewards):
    # play the best episode
    best_idx = np.argmax(training_rewards)
    best_episode = []
    for item in memory:
        if item[0] == best_idx:
            best_episode.append(item)
    
    best_actions = []
    for i in range(len(best_episode)):
        best_actions.append(best_episode[i][2])
    
    env.reset()    
    for action in best_actions:
        env.render()
        env.step(action)
    env.close()
    return best_idx, best_episode, best_actions


# TRAINING
training_rewards = []
for episode in range(TRAIN_EPISODES): 
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        action = epsilon_policy(state)
        new_state, reward, done, info = env.step(action)
        if done:
            reward = -reward  # penalty for loosing
        new_state = np.reshape(new_state, [1, observation_space])
        memory.append((episode, state, action, reward, new_state, done))
        state = new_state
        
        if done:
            print("Episode: {}, Exploration: {:2.2f}, Score: {}".format(episode, exploration_rate, step))
            break
        
        experience_replay()
        
        exploration_rate *= EXPLORATION_DECAY     # Decrease exploration rate
        exploration_rate = max(EXPLORATION_MIN, exploration_rate)
        
    training_rewards.append(step)

plot_score(TRAIN_EPISODES, training_rewards)
best_idx, best_episode, best_actions = play_best(training_rewards)

# TESTING
for episode in range(TEST_EPISODES): 
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        env.render()
        action = greedy_policy(state)
        new_state, reward, done, info = env.step(action)
        new_state = np.reshape(new_state, [1, observation_space])
        state = new_state
        
        if done:
            print("Episode: " + str(episode) + ", score: " + str(step))
            break  
env.close()
