import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import time
import random

from tqdm import tqdm
from collections import deque

# 환경 생성
env = gym.make('CartPole-v1', render_mode='ansi')
action_size = env.action_space.n

print('obs shape : ', env.observation_space.shape)
print('action_size : ', action_size)

# 하이퍼파라미터
n_episode = 1000
e = 0.9
e_min = 0.1
e_decay = 0.999
lr = 0.01

d_factor = 0.9
batch_size = 32

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(action_size)
    ])

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=lr)
    )
    return model

main = build_model()
target = build_model()
target.set_weights(main.get_weights())
memory = deque(maxlen=2000)

def replay():
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    states = []
    targets = []
    
    for state, action, reward, new_state, done in minibatch:
        pred = main.predict(state[np.newaxis], verbose=0)[0]
        
        if done:
            pred[action] = reward
        else:
            next_pred = target.predict(new_state[np.newaxis], verbose=0)[0]
            pred[action] = reward + d_factor * np.argmax(next_pred)

        states.append(state)
        targets.append(pred)
        
    main.fit(np.array(states), np.array(targets), verbose=0)

rList = []

all_step = 0
for i in tqdm(range(n_episode)):
    state, _ = env.reset()
    done = False

    step = 0
    while not done:
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            pred = main.predict(state[np.newaxis], verbose=0)[0]
            action = np.argmax(pred)

        new_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        if done:
            reward = -100
        
        memory.append((state, action, reward, new_state, done))
        
        state = new_state
        step += 1
        all_step += 1

        if step >= 1000:
            break
        
        if all_step % 20 == 0:
            replay()
        
        if all_step % 100 == 0:
            target.set_weights(main.get_weights())
        
    print(f'episode : {i}, e : {e:.2f}, step : {step} / {all_step}, reward : {reward}, done : {done}')
    
    if e > e_min:
        e *= e_decay


env.close()
# 검증

env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
done = False

cur = time.time()
while not done:
    pred = main.predict(state[None,:], verbose=0)[0]
    action = np.argmax(pred)
    
    new_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    state = new_state
    
print('time diff : ', (time.time() - cur) / 1000, '초 유지')

env.close()