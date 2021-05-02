# snake-rl
This repo tries to solve snake game (10, 10) and (20, 20) in an efficient way with multiple envs and fp16 training
using A2C and PPO.

## Dependencies

Slightly modified version of tianshou to support fp16 training and custom networks:

    pip install git+https://github.com/realiti4/tianshou.git@master --upgrade
    
Gym-snake environment:

    pip install git+https://github.com/realiti4/Gym-Snake.git@master --upgrade
    
## Training
Simply run `main.py`

## Notes

* It takes around 4 hours to train a decent A2C that reaches average of 80 reward (max is 97 for 10, 10). Still experimenting to max it and make it stable.

* PPO is more stable, but it takes a little bit longer to learn.

* These results acquired with 256 env running simultaneously, so it requires many steps. I optimized the environment above so it wasn't a problem even on a single core. On Linux you can use SubprocVectorEnv instead of dummy one to utilize all cores. It shouldn't be a problem to run hundreds of envs at the same time.

* You can comment out onpolicy_trainer(...) section and enable loading to watch pretrained agents.