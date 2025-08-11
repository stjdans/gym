import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from collections import deque
import random

# 환경 생성
env = gym.make('CliffWalking-v1', render_mode='ansi')
state_size = env.observation_space.n
action_size = env.action_space.n

print('state_size : ', state_size)
print('action_size : ', action_size)

# 하이퍼파라미터 설정
n_episode = 10
discount_factor = 0.9
e = 0.9
e_min = 0.1
e_decay = 0.9
lr = 0.001
batch_size = 32

def onehot(x):
    return np.identity(state_size)[x:x+1][0]

def get_action(table):
    return np.random.choice(4) if np.max(table) == 0 else np.argmax(table)

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(state_size * 2, activation='relu'),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(state_size*2, activation='relu'),
        # keras.layers.Dropout(0.3),
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
rList = []

# 경험 학습
def reply():
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    
    states = []
    targets = []
    
    for state_onehot, action, reward, new_state_onehot, done in minibatch:
        pred = main.predict(state_onehot[np.newaxis, :], verbose=0)[0]
        if done:
            pred[action] = reward
        else:
            next_pred = target.predict(new_state_onehot[np.newaxis,:], verbose=0)[0]
            pred[action] = reward + discount_factor * np.max(next_pred)

        states.append(state_onehot)
        targets.append(pred)

    main.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

for i in range(n_episode):
    # 환경 초기화
    state, info = env.reset()
    
    done = False
    while not done:
        state_onehot = onehot(state)
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            pred = main.predict(state_onehot[np.newaxis, :], verbose=0)[0]
            # action = np.argmax(pred + np.random.random(action_size) * e)
            action = np.argmax(pred)
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            reward = 100
        elif reward == -100:
            pass
        elif state == new_state:
            reward = -2
            
        # 경험 저장
        memory.append((state_onehot, action, reward, onehot(new_state), done))
        
        # if reward == -100:
        #     break
        
        state = new_state
    
    rList.append(reward)
    
    if e > e_min:
        e *= e_decay
        
    if i % 2 == 0:
        reply()
    
    # 가중치 업데이트
    if i % 10 == 0:
        target.set_weights(main.get_weights())
        
    print(f'episode : {i}, e : {e}, reward : {reward}, Done : {done}')
    
# plt.bar(range(100), rList[-100:])
# plt.title(f'{rList[-100:].count(100) / 100}%')
# plt.show()

env.close()

# 검증

# 환경 생성
env = gym.make('CliffWalking-v1', render_mode='human')

n_episode = 50

rList = []
for i in range(n_episode):
    state, _ = env.reset()
    
    done = False
    while not done:
        state_onehot = onehot(state)
        pred = main.predict(state_onehot[np.newaxis, :])
        action = np.argmax(pred[0])
        print(f'pred : {pred}')
        new_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        
        state = new_state
        
    rList.append(reward)

plt.bar(range(n_episode), rList)
plt.title(f'검증 : {rList.count(0) / n_episode}%')
plt.show()

env.close()