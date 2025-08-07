import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from collections import deque

# 환경 생성
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic

# 하이퍼파라미터
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
episodes = 500

# 상태/행동 공간
state_size = env.observation_space.n  # 16
action_size = env.action_space.n      # 4

# 상태를 one-hot으로 변환
def one_hot(state):
    vec = np.zeros(state_size)
    vec[state] = 1.0
    return vec

# Q-Network 모델
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size)  # Q(s,a)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mse')
    return model

# 경험 저장 버퍼
memory = deque(maxlen=2000)

# 모델 생성
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())  # 초기화

# 경험 재생
def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    states = []
    targets = []

    for state, action, reward, next_state, done in minibatch:
        target = model.predict(np.array([state]), verbose=0)[0]
        if done:
            target[action] = reward
        else:
            next_q = target_model.predict(np.array([next_state]), verbose=0)[0]
            target[action] = reward + gamma * np.max(next_q)

        states.append(state)
        targets.append(target)

    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

# 학습 루프
for episode in range(episodes):
    state, _ = env.reset()
    state = one_hot(state)
    total_reward = 0

    done = False
    while not done:
        # ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(action_size)
        else:
            q_values = model.predict(np.array([state]), verbose=0)[0]
            action = np.argmax(q_values)

        next_state, reward, done, _, _ = env.step(action)
        next_state_onehot = one_hot(next_state)

        # 종료 상태에서의 보상 보정
        if done and reward == 0:
            reward = -1

        # 경험 저장
        memory.append((state, action, reward, next_state_onehot, done))
        state = next_state_onehot
        total_reward += reward

        # 학습
        replay()
        print(episode)

    # ε 감소
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 타겟 모델 동기화
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode {episode:3d}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
