import numpy as np

TRAIN_STEPS = 39936
path        = './results_ext/scenario_1/PPO/history_0.npz'

data        = dict(np.load(path))
total_steps = len(data['violation'])

train_viol  = data['violation'][:TRAIN_STEPS]
eval_viol   = data['violation'][TRAIN_STEPS:]

print(f"Total steps    : {total_steps}")
print(f"Train steps    : {len(train_viol)}")
print(f"Eval steps     : {len(eval_viol)}")
print(f"Train viol rate: {(train_viol > 0).mean()*100:.2f}%")
print(f"Eval viol rate : {(eval_viol > 0).mean()*100:.2f}%")