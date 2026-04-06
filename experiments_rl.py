import sys
sys.path.insert(0, '.')
sys.path.insert(0, './gym-ran_slice')

import os
from numpy.random import default_rng
from itertools import product
from scenario_creator import create_env, get_slice_layout
from wrapper import ReportWrapper
from burst_wrapper import BurstWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from heuristic import _slice_load_ext
import torch
import numpy as np

RUNS               = 20   
TRAIN_STEPS        = 39936
EVALUATION_STEPS   = 10500 
CONTROL_STEPS      = 60000
PENALTY            = 1000
VERBOSE            = False
STEP_VERBOSE       = False
STEP_VERBOSE_EVERY = 100

run_list = list(range(RUNS))
scenarios = [0]            

algorithms = {
    'PPO': PPO,
}

deterministic = {
    'PPO': True,
}


def _format_step(step, action, obs, reward, info,
                 n_embb, n_urllc, n_mmtc, n_prbs, algo_name):
    loads    = _slice_load_ext(obs, n_embb, n_urllc, n_mmtc)
    viols    = np.array(info['violations'], dtype=int)
    n_slices = n_embb + n_urllc + n_mmtc
    regime   = info.get('regime_name', 'NORMAL')

    print(f"\n[{algo_name}] Step {step:>5d} | reward: {reward:>9.1f} | "
          f"free PRBs: {n_prbs - int(action.sum()):>3d} | "
          f"violations: {int(info['total_violations'])} | regime: {regime}",
          flush=True)
    print(f"  {'Slice':<6} {'Type':<7} {'PRBs':>5} {'Load':>7} {'SLA':>5}",
          flush=True)
    print(f"  {'─'*35}", flush=True)

    for i in range(n_slices):
        if i < n_embb:
            stype = 'eMBB'
        elif i < n_embb + n_urllc:
            stype = 'URLLC'
        else:
            stype = 'mMTC'
        sla = '✗' if viols[i] else '✓'
        print(f"  {i:<6} {stype:<7} {int(action[i]):>5} "
              f"{loads[i]:>7.3f} {sla:>5}", flush=True)


class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm):
        self.scenario  = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm

        self.path = './results_ext/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print("Creation of the directory %s failed" % self.path)
            else:
                print("Successfully created the directory %s " % self.path)

        self.model_path = './trained_models_ext/scenario_{}/{}/'.format(
            scenario, algo_name)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print("Creation of the directory %s failed" % self.model_path)
            else:
                print("Successfully created the directory %s " % self.model_path)

    def evaluate(self, i):
        print(f"start: scenario {self.scenario} run {i} "
              f"algorithm {self.algo_name}", flush=True)

        rng       = default_rng(seed=i)
        rng_burst = default_rng(seed=i + 1000)
        torch.manual_seed(i)

        n_embb, n_urllc, n_mmtc = get_slice_layout(self.scenario)

        base_env, _, _, _ = create_env(
            rng, self.scenario, penalty=PENALTY
        )
        base_env = base_env.unwrapped
        n_prbs   = base_env.n_prbs

        print(f"  slices: {n_embb} eMBB + {n_urllc} URLLC + {n_mmtc} mMTC | " f"PRBs: {n_prbs}", flush=True)

        node_env = BurstWrapper(
            base_env,
            n_embb=n_embb,
            n_urllc=n_urllc,
            n_mmtc=n_mmtc,
            rng=rng_burst,
            penalty=PENALTY,
            verbose=False
        )

        node_env = ReportWrapper(
            node_env,
            steps=TRAIN_STEPS,
            control_steps=CONTROL_STEPS,
            env_id=i,
            path=self.path,
            verbose=VERBOSE
        )

        env   = make_vec_env(lambda: node_env, n_envs=1)
        model = self.algorithm(
            'MlpPolicy', env,
            verbose=0,
            n_steps=512,
            batch_size=128,
        )

        print(f"training for {TRAIN_STEPS} steps...", flush=True)
        model.learn(total_timesteps=TRAIN_STEPS)

        node_env.save_results()
        model_path = f"{self.model_path}{self.algo_name}_agent_{i}"
        model.save(model_path)

        node_env.set_evaluation(EVALUATION_STEPS)
        obs = node_env.obs
        det = deterministic[self.algo_name]

        action, state = model.predict(obs, deterministic=det)

        for step in range(EVALUATION_STEPS):
            action, state = model.predict(obs, state=state, deterministic=det)
            obs, reward, done, info = node_env.step(action)

            if STEP_VERBOSE and step % STEP_VERBOSE_EVERY == 0:
                _format_step(step, action, obs, reward, info,
                             n_embb, n_urllc, n_mmtc, n_prbs, self.algo_name)

        node_env.save_results()
        print(f"results saved to {node_env.file_path}", flush=True)


if __name__ == '__main__':
    for scenario, (alg_name, alg) in product(scenarios, algorithms.items()):
        evaluator = RLEvaluator(scenario, alg_name, alg)
        for run in run_list:
            evaluator.evaluate(run)