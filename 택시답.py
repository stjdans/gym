import gymnasium as gym
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_RANDOM_SEED = 42



def train_q_learning(
    env, 
    use_action_mask: bool = True,
    episodes: int = 5000,
    seed: int = BASE_RANDOM_SEED,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    q_table = None,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    
    # n_states = env.observation_space.n
    # n_action = env.action_space.n
    # q_table = np.zeros((n_states, n_action))
    
    episode_rewards = []
    
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action_mask = info['action_mask'] if use_action_mask else None
        
            # 랜덤
            if np.random.random() < epsilon:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, n_action)
            
            # grredy                    
            else:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[np.argmax(q_table[state, valid_actions])]
                    else:
                        action = np.random.randint(0, n_action)
                
                else:
                    action = np.argmax(q_table[state])
                    
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Q-Learning update
            if not (done or truncated):
                if use_action_mask:
                    next_mask = info['action_mask']
                    valid_next_action = np.nonzero(next_mask == 1)[0]
                    if len(valid_actions) > 0:
                        next_max = np.max(q_table[next_state, valid_next_action])
                    else:
                        next_mask = 0
                
                else:
                    next_max = np.max(q_table[next_state])

                q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_factor * next_max - q_table[state, action]
                )
                
            state = next_state
        
        episode_rewards.append(total_reward)

    return {
        'episode_rewards': episode_rewards,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }
    

n_runs = 1
episodes = 5000
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1

seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]

masked_results_list = []
unmasked_results_list = []


q_table = None
un_q_table = None
for i, seed in enumerate(seeds):
    print(f'Run {i+1}/{n_runs} with seed {seed}')
    
    env_masked = gym.make('Taxi-v3')
    n_states = env_masked.observation_space.n
    n_action = env_masked.action_space.n
    q_table = np.zeros((n_states, n_action))
    
    masked_results = train_q_learning(
        env_masked,
        use_action_mask=True,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        episodes=episodes,
        q_table=q_table
    )
    
    env_masked.close()
    masked_results_list.append(masked_results)

    env_unmasked = gym.make('Taxi-v3')
    n_states = env_unmasked.observation_space.n
    n_action = env_unmasked.action_space.n
    un_q_table = np.zeros((n_states, n_action))
    
    unmasked_results = train_q_learning(
        env_unmasked,
        use_action_mask=False,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        episodes=episodes,
        q_table=un_q_table
    )

    env_unmasked.close()
    unmasked_results_list.append(unmasked_results)
    
print(masked_results_list)
masked_mean_rewards = [r["mean_reward"] for r in masked_results_list]
print(masked_mean_rewards)

unmasked_mean_rewards = [r["mean_reward"] for r in unmasked_results_list]

masked_overall_mean = np.mean(masked_mean_rewards)
masked_overall_std = np.std(masked_mean_rewards)
unmasked_overall_mean = np.mean(unmasked_mean_rewards)
unmasked_overall_std = np.std(unmasked_mean_rewards)

plt.figure(figsize=(12, 8), dpi=100)

# for i, (masked_results, unmasked_results) in enumerate(
#     zip(masked_results_list, unmasked_results_list
#         )):
#     plt.plot(
#         masked_results['episode_rewards'],
#         label='With Action Masking' if i == 0 else None,
#         color='blue',
#         alpha=0.1
#     )
#     plt.plot(
#         unmasked_results['episode_rewards'],
#         label='Without Action Masking' if i == 0 else None,
#         color='red',
#         alpha=0.1
#     )

# masked_mean_curve = np.mean([r['episode_rewards'] for r in masked_results_list], axis=0)
# unmasked_mean_curve = np.mean(
#     [r['episode_rewards'] for r in unmasked_results_list], axis=0
# )

# plt.plot(
#     masked_mean_curve, label='With Action Masking (Mean)', color='blue', linewidth=2
# )

# plt.plot(
#     unmasked_mean_curve,
#     label='Without Action Masking (Mean)',
#     color='red',
#     linewidth=2
# )

# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Training Performance: Q-Learning with vs without Action Masking')
# plt.legend()
# plt.grid(True, alpha=0.3)

# savefig_folder = Path('_static/img/tutorials/')
# savefig_folder.mkdir(parents=True, exist_ok=True)
# plt.savefig(
#     savefig_folder / "taxi_v3_action_masking_comparison.png",
#     bbox_inches='tight',
#     dpi=150
# )
# plt.show()


env = gym.make('Taxi-v3', render_mode='human')
state, info = env.reset()

print(q_table)

done = False
while not done:
    action_mask = info['action_mask']
    valid_actions = np.nonzero(action_mask == 1)[0]
    print(f'valid_actions : {valid_actions}')
    print(f'q_table[state, valid_actions] : {q_table[state, valid_actions]}')
    print(f'np.argmax(q_table[state, valid_actions]) : {np.argmax(q_table[state, valid_actions])}')
    action = valid_actions[np.argmax(q_table[state, valid_actions])]


    print('action : ', action)
    new_state, reward, terminate, truncated, info = env.step(action)
    print(f'new_state : {new_state}, reward : {reward}, termi : {terminate}')
    done = terminate or truncated
    
    state = new_state
    
env.close()

    