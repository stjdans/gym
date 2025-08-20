from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool
    n_runs: int
    action_size: int
    state_size: int
    proba_frozen: float
    
params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9
)

rng = np.random.default_rng(params.seed)


# 환경
env = gym.make(
    'FrozenLake-v1',
    is_slippery=params.is_slippery,
    render_mode='rgb_array',
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    )
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)

print(f'Action size : {params.action_size}')
print(f'State size : {params.state_size}')

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()
    
    def update(self, state, action, reward, new_state):
        delta = (
            reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update
    
    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))
        
class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
    def choose_action(self, action_space, state, qtable):
        explor_exploit_tradeoff = rng.uniform(0, 1)
        
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()
        
        else:
            max_ids = np
        