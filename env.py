#network slicing environment
# Network slicing environment: 2 slices sharing a fixed resource pool

import gym
from gym import spaces
import numpy as np

class TwoSliceEnv(gym.Env):
    """A simple network slicing environment with 2 slices."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TwoSliceEnv, self).__init__()

        self.total_resource = 40  # Total resources available to both slices

        # Observation space: queue sizes, latencies, and current allocations
        # [queue_A, queue_B, latency_A, latency_B, alloc_A, alloc_B]
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(6,), dtype=np.float32
        )

        # Action space: fraction of total resource allocated to slice A (slice B gets 1 - alloc_A)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reset()  # initialize queues and allocations

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial queue sizes for both slices (simulates different starting loads)
        self.queue_A = np.random.randint(10, 50)
        self.queue_B = np.random.randint(10, 50)

        # Initial even allocation of resources
        self.alloc_A = 0.5
        self.alloc_B = 0.5

        # Return initial observation
        return self._get_obs(), {}

    def _get_obs(self):
        # Latency is proportional to queue size and inversely proportional to allocated resources
        latency_A = self.queue_A / max(self.total_resource * self.alloc_A, 1e-6)
        latency_B = self.queue_B / max(self.total_resource * self.alloc_B, 1e-6)

        # Observation vector combines queues, latencies, and allocations
        return np.array([
            self.queue_A,
            self.queue_B,
            latency_A,
            latency_B,
            self.alloc_A,
            self.alloc_B
        ], dtype=np.float32)

    def step(self, action):
        # Clip action to valid range [0,1] and update allocations
        self.alloc_A = float(np.clip(action[0], 0, 1))
        self.alloc_B = 1 - self.alloc_A  # slice B gets the remaining resources

        # Compute latencies based on current allocation
        latency_A = self.queue_A / max(self.total_resource * self.alloc_A, 1e-6)
        latency_B = self.queue_B / max(self.total_resource * self.alloc_B, 1e-6)

        # Reward: negative sum of latencies (we want to minimize latency)
        reward = - (latency_A + latency_B)

        # Update queues after serving traffic according to allocation
        # Add random new arrivals to simulate stochastic traffic
        self.queue_A = max(self.queue_A - int(self.total_resource * self.alloc_A), 0) + np.random.randint(0,5)
        self.queue_B = max(self.queue_B - int(self.total_resource * self.alloc_B), 0) + np.random.randint(0,5)

        # Episode terminates when both queues are empty
        done = self.queue_A == 0 and self.queue_B == 0

        # Info dictionary for monitoring additional metrics (latency)
        info = {
            "latencyA": latency_A,
            "latencyB": latency_B
        }

        # Return next observation, reward, done flag, and info
        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        # Simple console rendering of queues and allocations
        print(f"QueueA: {self.queue_A}, QueueB: {self.queue_B}, AllocA: {self.alloc_A:.2f}, AllocB: {self.alloc_B:.2f}")
