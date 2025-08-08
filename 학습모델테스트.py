import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

n_episodes = 10

# 환경생성
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

def onehot(x):
    return np.identity(16)[x:x+1]

model = keras.models.load_model('best_dqn_037.keras')
print(model)

rList = []
for i in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    done = False
    while not done:
        state_onehot = onehot(state)
        print('state_onehot : ', state_onehot)
        pred = model.predict(state_onehot)
        print('pred : ', pred)
        action = np.argmax(pred)
        print('action : ' , action)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
    
        total_reward += reward
        state = next_state
        
    rList.append(total_reward)
    

print(f'episode : {i}, 확률 : {rList.count(1) / len(rList)}')