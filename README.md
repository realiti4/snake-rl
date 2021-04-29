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
