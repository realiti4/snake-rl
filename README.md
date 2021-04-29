# snake-rl
Solving Snake game fast with A2C and PPO using multiple agents and fp16 training

## Dependencies

Slightly modified version of tianshou to support fp16 training and custom networks:

    pip install git+https://github.com/realiti4/tianshou.git@master --upgrade
    
Gym-snake environment:

    pip install git+https://github.com/realiti4/Gym-Snake.git@master --upgrade
    
## Training
Simply run `main.py`
