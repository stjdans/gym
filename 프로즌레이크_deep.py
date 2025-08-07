import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tqdm import tqdm
import random

# 환경생성
env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

# 하이퍼파라미터
n_episode = 500
discount_factor = 0.9
e = 1.
e_decay = 0.995
e_min = 0.1
batch_size = 32
lr = 0.1

rList = []
memory = []

def onehot(x):
    return np.identity(state_size)[x:x+1][0]

def buildModel():
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(action_size)
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


def reply():
    global memory
    if len(memory) < batch_size:
        return
    
    # minibatch = random.sample(memory, batch_size)
    minibatch = memory[:batch_size]
    memory = memory[batch_size:]
    
    states = []
    q_values = []
    
    for state, action, reward, next_state, done in minibatch:
        state_onehot = onehot(state)
        next_state_onehot = onehot(next_state)
        
        q = model.predict(state_onehot[np.newaxis], verbose=0)[0]
        if done:
            q[action] = reward
        else:
            q_next = target.predict(next_state_onehot[np.newaxis], verbose=0)[0]
            q[action] = reward + discount_factor * np.max(q_next)
        
        states.append(state_onehot)
        q_values.append(q)
        
        
    # print('len states : ', len(states))
    # print('len q_values : ', len(q_values))
    # 훈련            
    model.fit(np.array(states), np.array(q_values), verbose=0, epochs=1)
    
    
# 모델 생성
model = buildModel()
target = buildModel()
target.set_weights(model.get_weights())

for episode in tqdm(range(n_episode)):
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        state_onehot = onehot(state)
        q_values = model.predict(state_onehot[np.newaxis], verbose=0)[0]
        
        # action 선택 방벙 : egreedy 사용
        if np.random.rand(1) < e:
            # action = env.action_space.sample()
            action = np.random.randint(action_size)
        else: 
            action = np.argmax(q_values)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and reward == 0:
            reward = -1

        memory.append((state, action, reward, next_state, done))
        total_reward += reward 
        state = next_state
        
        reply()
        
    rList.append(total_reward)
    
    if e > e_min:
        e *= e_decay
        
    if episode % 10 == 0:
        # 가중치 설정
        target.set_weights(model.get_weights())

    print('episode : {}, e : {}, total_reward : {}'.format(episode, e, total_reward))
        

env.close()

print('확률 : ', np.mean(rList, dtype=np.float32), '%')
plt.bar(range(len(rList)), rList)
plt.title(f'prop : {np.mean(rList, dtype=np.float32)}%')
plt.show()
    
