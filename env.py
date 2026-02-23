#network slicing environment

import gym
from gym import spaces
import numpy as np

class TwoSliceEnv(gym.Env):
    """A simple network slicing environment with 2 slices."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TwoSliceEnv, self).__init__()

        self.total_resource = 40

        # Observation: queue sizes, latencies, allocations
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(6,), dtype=np.float32
        )

        # Action: fraction of resource to allocate to slice A (slice B = 1 - A)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queue_A = np.random.randint(10, 50)
        self.queue_B = np.random.randint(10, 50)
        self.alloc_A = 0.5
        self.alloc_B = 0.5
        return self._get_obs(), {}

    def _get_obs(self):
        latency_A = self.queue_A / max(self.total_resource * self.alloc_A, 1e-6)
        latency_B = self.queue_B / max(self.total_resource * self.alloc_B, 1e-6)
        return np.array([
            self.queue_A,
            self.queue_B,
            latency_A,
            latency_B,
            self.alloc_A,
            self.alloc_B
        ], dtype=np.float32)

    def step(self, action):
        self.alloc_A = float(np.clip(action[0], 0, 1))
        self.alloc_B = 1 - self.alloc_A

        # Compute latencies and reward
        latency_A = self.queue_A / max(self.total_resource * self.alloc_A, 1e-6)
        latency_B = self.queue_B / max(self.total_resource * self.alloc_B, 1e-6)
        reward = - (latency_A + latency_B)

        # Update queues with random arrivals
        self.queue_A = max(self.queue_A - int(self.total_resource * self.alloc_A), 0) + np.random.randint(0,5)
        self.queue_B = max(self.queue_B - int(self.total_resource * self.alloc_B), 0) + np.random.randint(0,5)

        # Episode done when both queues empty
        done = self.queue_A == 0 and self.queue_B == 0

        info = {
            "latencyA": latency_A,
            "latencyB": latency_B
        }

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"QueueA: {self.queue_A}, QueueB: {self.queue_B}, AllocA: {self.alloc_A:.2f}, AllocB: {self.alloc_B:.2f}")
