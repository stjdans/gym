import gymnasium as gym 

env = gym.make("LunarLander-v3", render_mode='human', continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

print(f'observation_space : {env.observation_space}')

print(f'action_space : {env.action_space}')
print(f'action_space : {env.action_space.shape}')

state, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    
    print(f'state : {state}')
    print(f'info : {info}')
    print(f'action : {action}')
    
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f'next_state : {next_state}')
    print(f'reward : {reward}')
    print(f'terminated : {terminated}')
    print(f'info : {info}')
    
    
    done = terminated or truncated
    
    state = next_state
    
env.close()