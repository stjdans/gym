import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from collections import deque

# 환경 생성
env = gym.make('', render_mode='human')
state_size = env.observation_space.n
action_size = env.action_space.n

print('state_size : ', state_size)
print('action_size : ', action_size)

# 하이퍼파라미터
n_episode = 1
e = 0.9
e_min = 0.1
e_decay = 0.99
lr = 0.01

for i in range(n_episode):
    state, _ = env.reset()

    done = False

    while not done:
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, _ = env.step(action)

        state = new_state

    if e > e_min:
        e *= e_decay

env.close()