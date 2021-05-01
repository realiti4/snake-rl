import time
import numpy as np
import pandas as pd

import gym
import gym_snake

from wrapper_test import wrap_deepmind
import matplotlib.pyplot as plt

def make_env():
    return wrap_deepmind('snake-v0', frame_stack=3, scale=True, episode_life=False, clip_rewards=False, unit_size=2)


env = gym.make('snake-v0', unit_size=4)
env = make_env()

obs = env.reset()

start = time.time()
for i in range(80000):
    action = input('Where: ')
    # action = 1

    obs, reward, done, info = env.step(action)
    # obs = np.divide(obs, 255, dtype=np.float32)     # preprocess
    # print(done)
    env.render()
    # time.sleep(2)
    if done:
        obs = env.reset()
        # time.sleep(60)
        # break

print(f'It took {time.time()-start}')