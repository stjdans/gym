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
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            # print('max_ids : ', max_ids)
            action = rng.choice(max_ids)
            
        return action
        
        
learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size
)

explorer = EpsilonGreedy(epsilon=params.epsilon)


def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []
    
    for run in range(params.n_runs):
        learner.reset_qtable()
        
        for episode in tqdm(episodes, desc=f'Run {run}/{params.n_runs} - Episodes', leave=False):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0
            
            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )
                
                all_states.append(state)
                all_actions.append(action)
                
                new_state, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                
                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )
                
                total_rewards += reward
                step += 1
                
                state = new_state
                
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable
        
    return rewards, steps, episodes, qtables, all_states, all_actions    


def postprocess(episodes, params, rewards, steps, map_size):
    res = pd.DataFrame(
        data={
            'Episodes': np.tile(episodes, reps=params.n_runs),
            'Rewards': rewards.flatten(order='F'),
            'Steps': steps.flatten(order='F')
        }
    )
    
    
    res['cum_rewards'] = rewards.cumsum(axis=0).flatten(order='F')
    res['map_size'] = np.repeat(f'{map_size}x{map_size}', res.shape[0])
    print('res DF\n', res)
    
    st = pd.DataFrame(data={'Episodes': episodes, 'Steps': steps.mean(axis=1)})
    st['map_size'] = np.repeat(f'{map_size}x{map_size}', st.shape[0])
    
    print('st DF\n', st)
    return res, st

def qtable_directions_map(qtable, map_size):
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    qtable_val_max, qtabel_directions = qtable_directions_map(qtable, map_size)
    
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].imshow(env.render())
    ax[0].axis('off')
    ax[0].set_title('Last frame')
    
    sns.heatmap(
        qtable_val_max,
        annot=qtabel_directions,
        fmt='',
        ax=ax[1],
        cmap=sns.color_palette('Blues', as_cmap=True),
        linewidths=0.7,
        linecolor='black',
        xticklabels=[],
        yticklabels=[],
        annot_kws={'fontsize': 'xx-large'}
    ).set(title='Learned Q-values\nArrows represent best action')
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7),
        spine.set_color('black')
        
    plt.show()
    
def plot_states_actions_distribution(states, actions, map_size):
        labels = {'LEFT': 0, 'DOWN':1, 'RIGHT':2, 'UP':3}
        
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        sns.histplot(data=states, ax=ax[0], kde=True)
        ax[0].set_title('States')
        sns.histplot(data=actions, ax=ax[1])
        ax[1].set_xticks(list(labels.values()), labels=labels.keys())
        ax[1].set_title('Actions')
        fig.tight_layout()
        plt.show()

map_sizes = [4,7,9,11]
# map_sizes = [11]

res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        'FrozenLake-v1',
        is_slippery=params.is_slippery,
        render_mode='rgb_array',
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        )
    )
    
    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(params.seed)
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size
    )
    explore= EpsilonGreedy(epsilon=params.epsilon)
    
    print(f'Map size: {map_size}x{map_size}')
    rewards, steps, episodes, qtables, all_states, all_actions = run_env()
    
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)

    # plot_states_actions_distribution(
    #     states=all_states, actions=all_actions, map_size=map_size
    # )
    # plot_q_values_map(qtable, env, map_size)
    
    env.close()
    
def plot_steps_and_rewards(rewards_df, steps_df):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    sns.lineplot(
        data=rewards_df, x='Episodes', y='cum_rewards', hue='map_size', ax=ax[0]
    )
    ax[0].set(ylabel='Cumulated rewards')
    
    sns.lineplot(data=steps_df, x='Episodes', y='Steps', hue='map_size', ax=ax[1])
    ax[1].set(ylabel='Averaged steps number')
    
    for axi in ax:
        axi.legend(title='map size')
    fig.tight_layout()
    plt.show()
        
plot_steps_and_rewards(res_all, st_all)