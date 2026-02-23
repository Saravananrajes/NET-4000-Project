#simple baseline
import numpy as np
from env import TwoSliceEnv  # Custom environment for managing 2 network slices

# Initialize the environment
env = TwoSliceEnv()

print("Running Heuristic Baseline...\n")

# Reset the environment to get the initial observation
obs, _ = env.reset()  # obs is assumed to contain the queue sizes for Slice A and B
done = False
step = 0

while not done and step < 200:
    # Heuristic: allocate resources proportionally to queue sizes
    queue_A, queue_B = obs[0], obs[1]  # extract queue sizes from observation
    total = queue_A + queue_B + 1e-6  # small epsilon to prevent division by zero
    action = np.array([queue_A / total])  # allocate fraction of resources to Slice A; Slice B gets the rest

    # Take the action in the environment
    obs, reward, done, info = env.step(action)

    # Extract latency info for monitoring
    latA = info["latencyA"]
    latB = info["latencyB"]

    # Print step info
    print(f"Step {step} | LatA: {latA:.2f} | LatB: {latB:.2f} | Reward: {reward:.2f}")

    step += 1
