import gymnasium as gym
import numpy as np
from GridWorldEnv import GridWorldEnv
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FlattenObservation

gym.register(
    id='gym/GridWorld-v0',
    entry_point=GridWorldEnv,
    max_episode_steps=500
)

env = gym.make('gym/GridWorld-v0')
# check_env(env)

# print(env.unwrapped.size)
# obs, info = env.reset(seed=42)

# print('obs : ', obs)
# print('info : ', info)

# actions = [0, 1, 2, 3]
# for action in actions:
#     old_pos = obs['agent'].copy()
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     new_pos = obs['agent']
#     print(f'Action {action}: {old_pos} -> {new_pos}, reward = {reward}')


print(env.observation_space)

obs, info = env.reset()
env.render()
print(obs)

wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space)

obs, info = wrapped_env.reset()
print(obs)

wrapped_env.render()