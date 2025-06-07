import cv2
import gymnasium as gym
import numpy as np
from collections import deque


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.
    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(reward)

class PunishStuckAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Terminate or punish stuck agents
        self.count = 0
        self.punish_len = 4000
        self.max_len = 10000

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Max length control - terminate after not seeing a reward after n steps
        if reward == 0:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.punish_len:
            if self.count % 2000 == 0:    
                reward = -1    # We gonna start giving -1
            if self.count >= self.max_len:  # And we gonna terminate
                print('Debug: Reached max length')
                done = True
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.count = 0
        return obs, info

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.
    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=env.observation_space.shape,
            dtype=np.float32)

    def observation(self, observation):
        return (observation - self.bias) / self.scale

class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.
    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size), dtype=env.observation_space.dtype)

    def observation(self, frame):
        """returns the current observation from a frame"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame    # Don't resize for now
        return cv2.resize(frame, (self.size, self.size),
                          interpolation=cv2.INTER_AREA)

    # def step(self, action):
    #     obs, reward, done, truncated, info = self.env.step(action)
    #     return self.observation(obs), reward, done, info

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.
    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (n_frames,) + env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)

def wrap_deepmind(env_id, episode_life=False, clip_rewards=True,
                  frame_stack=4, scale=False, warp_frame=True,
                  punish_stuck_agent=False, **kwargs):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).
    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    # assert 'NoFrameskip' in env_id
    env = gym.make(env_id, **kwargs)
    # env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)
    # if episode_life:
    #     env = EpisodicLifeEnv(env)  # Commented out as EpisodicLifeEnv is not defined
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if punish_stuck_agent:
        env = PunishStuckAgent(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env