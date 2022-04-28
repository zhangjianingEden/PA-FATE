from util import *


class ShareRolloutStorage:
    def __init__(self):
        self.buffer = deque()

    def reset(self):
        self.buffer.clear()

    def push(self, obs, action, reward, next_obs, done):
        obs = obs.unsqueeze(0)
        next_obs = next_obs.unsqueeze(0)
        self.buffer.append((obs, action, reward, next_obs, done))
