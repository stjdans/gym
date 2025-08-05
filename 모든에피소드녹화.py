import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import logging

training_period = 250
num_eval_episodes = 10_000
env_name = 'CartPole-v1'

logging.basicConfig(level=logging.INFO, format='%(message)s')

env = gym.make(env_name, render_mode='rgb_array')

env = RecordVideo(
    env,
    video_folder='cartpole-training',
    name_prefix='training',
    episode_trigger=lambda x: x % training_period == 0
)

env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

print(f'Starting evaluation for {num_eval_episodes} ...')
print(f"Videos will be recorded every {training_period} episodes")
print(f'Videos will be saved to: cartpole-training/')

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count +=1
        
        episode_over = terminated or truncated
        
    # print(f'Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}')
  
    # if 'episode' in info:
    #     episode_data = info['episode']
    #     logging.info(f'Episode {episode_num}: '
    #                  f'reward={episode_data["r"]:.1f} '
    #                  f'length={episode_data["l"]} '
    #                  f'time={episode_data["t"]:.2f}s')
        
        if episode_num % 1000 == 0:
            recent_reward = list(env.return_queue)[-100:]
            if recent_reward:
                avg_recent = sum(recent_reward) / len(recent_reward)
                print(f'  -> Average reward over last 100 episodes: {avg_recent:.1f}')
            
env.close()

# print(f'Evaluation Summary:')
# print(f'Episode durations: {list(env.time_queue)}')
# print(f'Episode reward: {list(env.return_queue)}')
# print(f'Episode lengths: {list(env.length_queue)}')

# avg_reward = np.mean(env.return_queue)
# avg_length = np.mean(env.length_queue)
# std_reward = np.std(env.return_queue)

# print(f'Average reward: {avg_reward:.2f} +- {std_reward:.2f}')
# print(f'Average episode length: {avg_length:.1f} steps')
# print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')