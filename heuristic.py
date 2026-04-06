import numpy as np
import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './gym-ran_slice')

EMBB_N_VARS = 10
MMTC_N_VARS = 3

EMBB_CBR_TH    = 1
EMBB_CBR_QUEUE = 3
EMBB_VBR_TH    = 6
EMBB_VBR_QUEUE = 8
MMTC_DEVICES   = 0
MMTC_DELAY     = 2


def _env_params_ext(env, n_embb, n_urllc, n_mmtc):
    n_prbs   = int(env.n_prbs)
    n_slices = int(env.n_slices)
    return n_prbs, n_slices, n_embb, n_urllc, n_mmtc


def _slice_load_ext(obs, n_embb, n_urllc, n_mmtc):

    loads      = []
    n_embb_all = n_embb + n_urllc

    for k in range(n_embb_all):
        base            = k * EMBB_N_VARS
        cbr_th_pressure = obs[base + EMBB_CBR_TH]
        vbr_th_pressure = obs[base + EMBB_VBR_TH]
        cbr_q           = obs[base + EMBB_CBR_QUEUE]
        vbr_q           = obs[base + EMBB_VBR_QUEUE]
        load = 0.6 * (cbr_th_pressure + vbr_th_pressure) / 2.0 \
             + 0.4 * (cbr_q + vbr_q) / 2.0
        loads.append(min(load, 2.0))

    embb_offset = n_embb_all * EMBB_N_VARS
    for k in range(n_mmtc):
        base    = embb_offset + k * MMTC_N_VARS
        devices = obs[base + MMTC_DEVICES]
        delay   = obs[base + MMTC_DELAY]
        loads.append(min(0.4 * devices + 0.6 * delay, 2.0))

    return np.array(loads, dtype=float)


def _violation_flags(info):
    return np.array(info['violations'], dtype=bool)


def _distribute_prbs(weights, n_prbs, n_slices, min_prbs=1):
    weights  = np.maximum(weights, 1e-9)
    weights /= weights.sum()
    alloc    = np.floor(weights * n_prbs).astype(int)
    alloc    = np.maximum(alloc, min_prbs)
    remainder = n_prbs - alloc.sum()
    if remainder > 0:
        fracs = (weights * n_prbs) - np.floor(weights * n_prbs)
        alloc[np.argsort(fracs)[::-1][:remainder]] += 1
    if alloc.sum() > n_prbs:
        margin = alloc - min_prbs
        excess = alloc.sum() - n_prbs
        for _ in range(excess):
            idx = np.argmax(margin)
            alloc[idx]  -= 1
            margin[idx] -= 1
    return alloc


def _format_step_ext(step, action, obs, reward, info, n_embb, n_urllc, n_mmtc, n_prbs):
    loads    = _slice_load_ext(obs, n_embb, n_urllc, n_mmtc)
    viols    = np.array(info['violations'], dtype=int)
    n_slices = n_embb + n_urllc + n_mmtc
    regime   = info.get('regime_name', 'NORMAL')

    print(f"\nStep {step:>4d} | reward: {reward:>9.1f} | " f"free PRBs: {n_prbs - int(action.sum()):>3d} | " f"violations: {int(info['total_violations'])} | regime: {regime}")
    print(f"  {'Slice':<6} {'Type':<7} {'PRBs':>5} {'Load':>7} {'SLA':>5}")
    print(f"  {'─'*35}")

    for i in range(n_slices):
        if i < n_embb:
            stype = 'eMBB'
        elif i < n_embb + n_urllc:
            stype = 'URLLC'
        else:
            stype = 'mMTC'
        sla = '✗' if viols[i] else '✓'
        print(f"  {i:<6} {stype:<7} {int(action[i]):>5} {loads[i]:>7.3f} {sla:>5}")


class AIMDAgentExtended:
    def __init__(self, env, n_embb, n_urllc, n_mmtc, step_up=None, factor_down=0.95, min_prbs=2, urllc_min_frac=0.15):
        self.n_prbs   = int(env.n_prbs)
        self.n_slices = int(env.n_slices)
        self.n_embb   = n_embb
        self.n_urllc  = n_urllc
        self.n_mmtc   = n_mmtc

        self.step_up      = step_up if step_up is not None \
                            else max(1, int(0.25 * self.n_prbs / self.n_slices))
        self.factor_down  = factor_down
        self.min_prbs     = min_prbs
        self.urllc_min_frac = urllc_min_frac

        self._urllc_floor = max(
            int(urllc_min_frac * self.n_prbs / max(n_urllc, 1)),
            min_prbs
        )

        self._targets    = np.full(self.n_slices, self.n_prbs / self.n_slices)
        self._violations = np.zeros(self.n_slices, dtype=bool)
        self._regime     = 0 

    def predict(self, obs, state=None, deterministic=True):
        urllc_start = self.n_embb
        urllc_end   = self.n_embb + self.n_urllc

        step_up = self.step_up
        if self._regime == 1:  
            step_up = self.step_up * 2

        for k in range(self.n_slices):
            if self._violations[k]:
                self._targets[k] += step_up
            else:
                self._targets[k] = max(
                    self._targets[k] * self.factor_down,
                    float(self.min_prbs)
                )
        for k in range(urllc_start, urllc_end):
            self._targets[k] = max(self._targets[k], float(self._urllc_floor))

        alloc = _distribute_prbs(
            self._targets.copy(), self.n_prbs, self.n_slices, self.min_prbs
        )

        for k in range(urllc_start, urllc_end):
            if alloc[k] < self._urllc_floor:
                deficit = self._urllc_floor - alloc[k]
                alloc[k] += deficit
                non_urllc = [i for i in range(self.n_slices) if i < urllc_start or i >= urllc_end]
                for _ in range(deficit):
                    idx = max(non_urllc, key=lambda i: alloc[i] - self.min_prbs)
                    if alloc[idx] > self.min_prbs:
                        alloc[idx] -= 1

        self._targets = alloc.astype(float)
        return alloc, state

    def update(self, info):
        self._violations = _violation_flags(info)
        self._regime     = info.get('regime', 0)

    def reset(self):
        self._targets    = np.full(self.n_slices, self.n_prbs / self.n_slices)
        self._violations = np.zeros(self.n_slices, dtype=bool)
        self._regime     = 0


def evaluate_agent_ext(env, agent, n_embb, n_urllc, n_mmtc,
                       n_episodes=1, max_steps=10500, verbose=True,
                       step_verbose=False, step_verbose_every=1,
                       save_path=None):

    episode_rewards      = []
    all_viol_flags       = []
    all_viol_counts      = []
    all_free_prbs        = []
    per_slice_violations = np.zeros(env.n_slices, dtype=int)
    total_steps          = 0

    all_rewards    = []
    all_violations = []
    all_resources  = []

    regime_viol    = {0: 0, 1: 0, 2: 0}
    regime_steps   = {0: 0, 1: 0, 2: 0} 

    n_prbs = int(env.n_prbs)

    for ep in range(n_episodes):
        obs = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()

        ep_reward = 0.0

        if step_verbose:
            print(f"\n{'═'*55}")
            print(f"  Episode {ep + 1}")
            print(f"{'═'*55}")

        for step in range(max_steps):
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)

            if hasattr(agent, 'update'):
                agent.update(info)

            ep_reward   += reward
            total_steps += 1

            viol_arr = np.array(info['violations'], dtype=int)
            per_slice_violations += viol_arr
            n_viol = int(viol_arr.sum())
            all_viol_flags.append(1 if n_viol > 0 else 0)
            all_viol_counts.append(n_viol)

            all_rewards.append(reward)
            all_violations.append(1 if n_viol > 0 else 0)
            all_resources.append(int(action.sum()))

            regime = info.get('regime', 0)
            regime_steps[regime] += 1
            if n_viol > 0:
                regime_viol[regime] += 1

            if n_viol == 0:
                all_free_prbs.append(max(0, n_prbs - int(action.sum())))

            if step_verbose and step % step_verbose_every == 0:
                _format_step_ext(step, action, obs, reward, info,
                                 n_embb, n_urllc, n_mmtc, n_prbs)

            if done:
                break

        episode_rewards.append(ep_reward)
        if verbose:
            ep_vr = np.mean(all_viol_flags[-max_steps:]) * 100
            print(f"Episode {ep+1:3d} | reward: {ep_reward:10.1f} | "
                  f"SLA violation rate: {ep_vr:.1f}%")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path,
                 reward    = np.array(all_rewards,    dtype=np.float64),
                 violation = np.array(all_violations, dtype=np.int16),
                 resources = np.array(all_resources,  dtype=np.int16))
        print(f"Saved history to {save_path}")

    def regime_viol_rate(r):
        return regime_viol[r] / max(regime_steps[r], 1)

    results = dict(
        mean_reward              = float(np.mean(episode_rewards)),
        sla_violation_rate       = float(np.mean(all_viol_flags)),
        mean_violations_step     = float(np.mean(all_viol_counts)),
        mean_free_prbs           = float(np.mean(all_free_prbs)) if all_free_prbs else 0.0,
        per_slice_violation_rate = (per_slice_violations / max(total_steps, 1)).tolist(),
        violation_rate_normal    = float(regime_viol_rate(0)),
        violation_rate_burst     = float(regime_viol_rate(1)),
        violation_rate_lull      = float(regime_viol_rate(2)),
        episode_rewards          = episode_rewards,
    )

    if verbose:
        print(f"  Mean reward          : {results['mean_reward']:.2f}")
        print(f"  SLA violation rate   : {results['sla_violation_rate']*100:.2f}%")
        print(f"  Mean violations/step : {results['mean_violations_step']:.3f}")
        print(f"  Mean free PRBs       : {results['mean_free_prbs']:.2f}")
        print(f"  Viol rate (NORMAL)   : {results['violation_rate_normal']*100:.2f}%")
        print(f"  Viol rate (BURST)    : {results['violation_rate_burst']*100:.2f}%")
        print(f"  Viol rate (LULL)     : {results['violation_rate_lull']*100:.2f}%")
        print(f"  Per-slice viol rate  : "f"{[f'{v*100:.1f}%' for v in results['per_slice_violation_rate']]}")

    return results


if __name__ == '__main__':
    from numpy.random import default_rng
    from scenario_creator import create_env, get_slice_layout
    from burst_wrapper import BurstWrapper

    PENALTY    = 1000
    N_EPISODES = 1
    MAX_STEPS  = 10500
    SEED       = 42

    all_results = {}

    for SCENARIO in [0, 1, 2, 3]:
        n_embb, n_urllc, n_mmtc = get_slice_layout(SCENARIO)

        print(f"AIMD Scenario {SCENARIO}")

        rng       = default_rng(seed=SEED)
        rng_burst = default_rng(seed=SEED + 1000)

        base_env, _, _, _ = create_env(rng, SCENARIO, penalty=PENALTY)
        base_env = base_env.unwrapped

        env = BurstWrapper(
            base_env,
            n_embb=n_embb,
            n_urllc=n_urllc,
            n_mmtc=n_mmtc,
            rng=rng_burst,
            penalty=PENALTY,
            verbose=False
        )

        agent = AIMDAgentExtended(env, n_embb, n_urllc, n_mmtc)

        results = evaluate_agent_ext(
            env, agent, n_embb, n_urllc, n_mmtc,
            n_episodes=N_EPISODES,
            max_steps=MAX_STEPS,
            step_verbose=False,
            step_verbose_every=100,
            save_path=f'./results_ext/scenario_{SCENARIO}/AIMD/history_0.npz'
        )

        all_results[SCENARIO] = results

    print("  Final Comparison — All Extended Scenarios")
    print(f"{'Scenario':<10} {'Reward':>10} {'Viol%':>8} "f"{'Normal%':>9} {'Burst%':>8} {'Lull%':>7}")
    for sc, r in all_results.items():
        print(f"{sc:<10} {r['mean_reward']:>10.1f} "
              f"{r['sla_violation_rate']*100:>7.2f}% "
              f"{r['violation_rate_normal']*100:>8.2f}% "
              f"{r['violation_rate_burst']*100:>7.2f}% "
              f"{r['violation_rate_lull']*100:>6.2f}%")