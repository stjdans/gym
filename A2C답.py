from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import gymnasium as gym

class A2C(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        ciritic_lr: float,
        actor_lr: float,
        n_envs: int
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        
        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ]
        
        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        ]
        
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=ciritic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
        
        
    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)
        action_logits_vec = self.actor(x)
        return (state_values, action_logits_vec)
    
    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        state_value, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_value, entropy
    
    def get_losses(
        self, 
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)
        
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t+1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae
            
        critic_loss = advantages.pow(2).mean()
        
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        
        return (critic_loss, actor_loss)
        
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

n_envs = 10
n_updates = 1000
n_steps_per_update = 128
randomize_domain = False

gamma = 0.999
lam = 0.95
ent_coef = 0.01
actor_lr = 0.001
critic_lr = 0.005


if randomize_domain:
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                'LunarLander-v3',
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=1.0), a_min=-11.9, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=600
            )
            for i in range(n_envs)
        ]
    )
    
else:
    envs = gym.make_vec('LunarLander-v3', num_envs=n_envs, max_episode_steps=600)
    
obs_shape = envs.single_observation_space.shape[0]
action_shape = envs.single_action_space.n

use_cuda = False
if use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
    envs, buffer_length=n_envs * n_updates
)

critic_losses = []
actor_losses = []
entropies = []

for sample_phase in tqdm(range(n_updates)):
    ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
    masks = torch.zeros(n_steps_per_update, n_envs, device=device)
    
    if sample_phase == 0:
        states, info = envs_wrapper.reset(seed=42)
        
    for step in range(n_steps_per_update):
        actions, actions_log_probs, state_value_preds, entropy = agent.select_action(
            states
        )
        
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions.cpu().numpy()
        )
        
        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = torch.tensor(rewards, device=device)
        ep_action_log_probs[step] = actions_log_probs
        
        masks[step] = torch.tensor([not term for term in terminated])
        
    
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        entropy,
        masks,
        gamma,
        lam,
        ent_coef,
        device
    )
    
    agent.update_parameters(critic_loss, actor_loss)
    
    critic_losses.append(critic_loss.detach().cpu().numpy())
    actor_losses.append(actor_loss.detach().cpu().numpy())
    entropies.append(entropy.detach().mean().cpu().numpy())
    
    #plot
    rolling_length = 20
    