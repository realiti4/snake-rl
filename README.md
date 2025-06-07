# snake-rl
This repo tries to solve snake game (10, 10) and (20, 20) in an efficient way with multiple envs and fp16 training
using A2C and PPO.

## Installation

This project uses `uv` for dependency management and supports modern Python package standards via `pyproject.toml`.

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Install with uv (Recommended)
```bash
# Install uv if you haven't already
pip install uv

# Install the project and all dependencies
uv sync
```

### Install with pip (Alternative)
```bash
pip install -e .
```

The project automatically installs:
- Modified version of tianshou with fp16 training and custom networks support
- Gym-Snake environment
- PyTorch with CUDA 12.8 support (configurable via tool.uv.sources in pyproject.toml)
    
## Training

With uv:
```bash
uv run main_a2c.py  # for A2C training
# or
uv run main_ppo.py  # for PPO training
```

With traditional Python:
```bash
python main_a2c.py  # for A2C training
# or
python main_ppo.py  # for PPO training
```

## Notes

* It takes around 4 hours to train a decent A2C that reaches average of 80 reward (max is 97 for 10, 10). Still experimenting to max it and make it stable.

* PPO is more stable, but it takes a little bit longer to learn.

* These results acquired with 256 env running simultaneously, so it requires many steps. I optimized the environment above so it wasn't a problem even on a single core. On Linux you can use SubprocVectorEnv instead of dummy one to utilize all cores. It shouldn't be a problem to run hundreds of envs at the same time.

* You can comment out onpolicy_trainer(...) section and enable loading to watch pretrained agents.