import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# import keyboard

# 환경 생성
env = gym.make('CliffWalking-v1', render_mode='ansi')
state_size = env.observation_space.n
action_size = env.action_space.n

# 하이퍼파라미터 설정
n_episode = 1000
discount_factor = 0.9
e = 0.9
e_min = 0.1
e_decay = 0.995

q_table = np.zeros((state_size, action_size), dtype=np.float64)
print(q_table)
print(q_table.shape)
print(q_table[0].shape)
print(q_table[0, :].shape)

def get_action(table):
    return np.random.choice(4) if np.max(table) == 0 else np.argmax(table)
    
print('state_size : ', state_size)
print('action_size : ', action_size)

rList = []

for i in range(n_episode):
    # 환경 초기화
    state, info = env.reset()
    
    done = False
    while not done:
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = get_action(q_table[state])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:    # 목적지에 도달한 경우
            reward = 100
        elif reward == -100:
            reward = -100
        elif state == new_state: # 다음 이동할 칸이 같은 위치인 경우
            reward = -2
        
        q_table[state, action] = reward + discount_factor * np.max(q_table[new_state])
        
        # if reward == -100:
        #     break
        
        state = new_state
    
    print(q_table)
    rList.append(reward)
    
    if e > e_min:
        e *= e_decay
        
    print(f'episode : {i}, e : {e}, reward : {reward}, Done : {done}')
    
# plt.bar(range(100), rList[-100:])
# plt.title(f'{rList[-100:].count(100) / 100}%')
# plt.show()

env.close()

# 검증

# 환경 생성
env = gym.make('CliffWalking-v1', render_mode='human')

n_episode = 100

rList = []
for i in range(n_episode):
    state, _ = env.reset()
    
    done = False
    while not done:
        action = np.argmax(q_table[state])
        print(f'q_table[{state}] : {q_table[state]}')
        new_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated or (reward == -100)
        
        state = new_state
        
    rList.append(reward)

plt.bar(range(n_episode), rList)
plt.title(f'검증 : {rList.count(0) / n_episode}%')
plt.show()

env.close()