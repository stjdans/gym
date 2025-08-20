import gymnasium as gym
import seaborn as sns
from collections import defaultdict
from BlackjackAgent import BlackjackAgent
from tqdm import tqdm
from matplotlib.patches import Patch

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

print('{:.5f}'.format(epsilon_decay))


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

def create_grids(agent, usable_ace=False):
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))
        
    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11)
    )
    
    print('player_count \n', player_count)
    
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count])
    )
    value_grid = player_count, dealer_count, value
    
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)
    
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap='viridis',
        edgecolor='none'
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)
    
    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)
    
        # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()