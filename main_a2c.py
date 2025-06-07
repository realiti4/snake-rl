import numpy as np
import gymnasium as gym
import gym_snake

import torch
import torch.nn as nn

import tianshou as ts
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from model.cnn_model import CNNModel
from wrapper_test import wrap_deepmind

def make_env_wrapper():
    return wrap_deepmind('snake-v0', frame_stack=3, scale=True, episode_life=False, clip_rewards=False, unit_size=4, unit_gap=0)

device = 'cuda'

make_env = lambda: gym.make('snake-v0')

lr, epoch, batch_size = 1e-3, 100, 256
train_num, test_num = 256, 16
buffer_size = 2000

# Use SubprocVectorEnv instead of Dummy one if you are on
train_envs = ts.env.DummyVectorEnv([lambda: make_env_wrapper() for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: make_env_wrapper() for _ in range(test_num)])

state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

# Actor, Critic
custom_kwargs = {'output_dim': 1024, 'kernel_size': 8, 'dropout': 0.25}
hidden_sizes=[1024, 1024, 1024]

net = Net(state_shape, hidden_sizes=hidden_sizes,
              device=device,
              custom_model=CNNModel,
              custom_model_kwargs=custom_kwargs,
              )
actor = Actor(net, action_shape, device=device).to(device)
critic = Critic(net, device=device).to(device)
optim = torch.optim.Adam        # dummy optim

# Setup policy and collectors
dist = torch.distributions.Categorical
policy = ts.policy.A2CPolicy(actor, critic, optim, dist,
            max_grad_norm=0.5,
            use_autocast=True,
            )

policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)

# Collector
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, buffer_num=train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs)

# Save and Load
load = True

def save_fn(policy, add=''):
    name = f'saved/snake_a2c{add}.pth'
    torch.save({'policy_state': policy.state_dict(),
                'optimizer_state': policy.optim.state_dict(),
                'scaler': policy.scaler.state_dict(),
                }, name
                )
if load:
    new_lr = 1e-3
    load_dict = torch.load('saved/snake_a2c.pth')
    policy.load_state_dict(load_dict['policy_state'])
    policy.optim.load_state_dict(load_dict['optimizer_state'])
    policy.scaler.load_state_dict(load_dict['scaler'])
    policy.optim.param_groups[0]['lr'] = new_lr
    print(f'Learning rate is set to = {new_lr}')


result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=epoch,
        step_per_epoch=1600000,
        repeat_per_collect=1,
        episode_per_test=test_num,
        batch_size=1024,
        step_per_collect=train_num*5,      # 5
        # episode_per_collect=10,
        save_fn=save_fn,
        # backup_save_freq=0,
        )

# Fun part see the results!
eval_env = make_env_wrapper()
eval_collector = ts.data.Collector(policy, eval_env)
policy.eval()
eval_collector.collect(n_episode=10, render=0.005)
