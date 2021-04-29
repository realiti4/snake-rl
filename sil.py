import time
import numpy as np
import pandas as pd

import gym
import gym_snake


env = gym.make('snake-v0', unit_size=10)

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