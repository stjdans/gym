import gymnasium as gym
from BlackjackAgent import BlackjackAgent
from tqdm import tqdm

learning_rate = 0.1
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon
final_epsilon = 0.1

env = gym.make('Blackjack-v1', sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    
    while not done:
        action = agent.get_action(obs)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        agent.update(obs, action, reward, terminated, next_obs)
        
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
    
    
    
from matplotlib import pyplot as plt
import numpy as np

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window
    
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title('rewards')
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    'valid'
)

axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel('Average Reward')
axs[0].set_xlabel('Episode')


axs[1].set_title('lengths')
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    'valid'
)

axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel('Average lengths')
axs[1].set_xlabel('Episode')

axs[2].set_title('Training Error')
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    'same'
)

axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel('Temporal Difference Error')
axs[2].set_xlabel('Step')


plt.tight_layout()
plt.show()


def test_agent(agent, env, num_episodes=1000):
    total_reward = []
    
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        total_reward.append(episode_reward)
        
    agent.epsilon = old_epsilon
    
    win_rate = np.mean(np.array(total_reward) > 0)
    average_reward = np.mean(total_reward)
    
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_reward):.3f}")
    
# Test your agent
test_agent(agent, env)

print('*' * 100)

print(agent.q_values)