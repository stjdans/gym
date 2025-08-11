import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import commonfuc as cf

discount_factor = 0.9
e = 0.99
n_episode = 2_000
lr = 0.8

env = gym.make('FrozenLake-v1', is_slippery=True)

print('observation_space ', env.observation_space.n)
print('action_space ', env.action_space.n)
print(env.action_space)


rList = []
Q = np.zeros([env.observation_space.n, env.action_space.n])
print('Q : \n', Q)

for episode in range(n_episode):
    obs, info = env.reset()
    done = False
    rewards = 0
    e -= ((episode+1) / 1000)
    
    while not done:
        # action 선택 방벙 : egreedy 사용
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else: 
        #     action = cf.rrand(Q[obs, :])
            
        action = np.argmax(Q[obs, :] + np.random.randn(1, 4) / (episode+1))
        
        # print('action : ', action)
        newObs, reward, terminated, truncated, info = env.step(action)
        # print(f'newObs = {newObs},\n reward = {reward},\n')
        
        # 현재 Q 테이블 업데이트
        # Q[obs, action] = reward + discount_factor * np.max(Q[newObs, :])
        Q[obs, action] = (1-lr)*Q[obs, action] + lr*(reward + discount_factor * np.max(Q[newObs, :]))
        
        rewards += reward
        done = terminated or truncated
        # if done and not reward:
        #     Q[obs, action] *= 0.99
        
        obs = newObs
    
    rList.append(rewards)

print('Q >>>\n', Q)
print('e : ', e)
print('확률 : ', np.mean(rList, dtype=np.float32), '%')

env.close()

plt.bar(range(len(rList)), rList)
plt.title(f'확률 : {np.mean(rList, dtype=np.float32)}%')
plt.show()
    
