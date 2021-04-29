import time
import numpy as np
import pandas as pd

import gym
import gym_snake

from wrapper_test import wrap_deepmind

def make_env():
    return wrap_deepmind('snake-v0', frame_stack=3, scale=True, episode_life=False, warp_frame=False)


env = gym.make('snake-v0', unit_size=10)
env = make_env()

obs = env.reset()

for i in range(1000):
    action = input('Where: ')

    obs, reward, done, info = env.step(action)
    obs = np.divide(obs, 255, dtype=np.float32)     # preprocess
    print(done)
    env.render()
    # time.sleep(2)
    # if done:
        
    #     # time.sleep(60)
    #     break

print('de')