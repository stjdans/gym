import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

n_episodes = 100

# 환경생성
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

def onehot(x):
    return np.identity(16)[x:x+1]

model = keras.models.load_model('best_dqn_028_s.keras')

rList = []
for i in range(n_episodes):
    state, _ = env.reset()
    
    done = False
    while not done:
        state_onehot = onehot(state)
        pred = model.predict(state_onehot, verbose=0)
        action = np.argmax(pred)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
    
        state = next_state
        
    rList.append(reward)
    

print(f'episode : {i}, 확률 : {rList.count(1) / len(rList)}')