#simple baseline
import numpy as np
from env import TwoSliceEnv

env = TwoSliceEnv()

print("Running Heuristic Baseline...\n")

obs, _ = env.reset()
done = False
step = 0

while not done and step < 200:
    # Heuristic: allocate proportionally to queue sizes
    queue_A, queue_B = obs[0], obs[1]
    total = queue_A + queue_B + 1e-6
    action = np.array([queue_A / total])  # allocate fraction to A

    obs, reward, done, info = env.step(action)

    latA = info["latencyA"]
    latB = info["latencyB"]

    print(f"Step {step} | LatA: {latA:.2f} | LatB: {latB:.2f} | Reward: {reward:.2f}")

    step += 1
