import numpy as np
import gym

NORMAL = 0
BURST  = 1
LULL   = 2

STATE_NAMES = {NORMAL: 'NORMAL', BURST: 'BURST', LULL: 'LULL'}

TRANSITION = {
    NORMAL: {NORMAL: 0.992, BURST: 0.005, LULL: 0.003},
    BURST:  {NORMAL: 0.05,  BURST: 0.95,  LULL: 0.0},
    LULL:   {NORMAL: 0.03,  BURST: 0.0,   LULL: 0.97},
}


REGIME_PENALTY_SCALE = {
    NORMAL: 1.0,
    BURST:  2.5,  
    LULL:   0.5,
}

REGIME_EFFICIENCY_SCALE = {
    NORMAL: 1.0,
    BURST:  0.5,  
    LULL:   1.5,   
}

URLLC_BURST_EXTRA_PENALTY = 1.5


class BurstWrapper(gym.Wrapper):
    N_REGIME_FEATURES = 3

    def __init__(self, env, n_embb, n_urllc, n_mmtc, rng, penalty=1000, verbose=False):
        super().__init__(env)
        self.n_embb   = n_embb
        self.n_urllc  = n_urllc
        self.n_mmtc   = n_mmtc
        self.rng      = rng
        self.penalty  = penalty
        self.verbose  = verbose

        self._state      = NORMAL
        self._step_count = 0

        self._urllc_start = n_embb
        self._urllc_end   = n_embb + n_urllc

    def _transition(self):
        probs  = TRANSITION[self._state]
        states = [NORMAL, BURST, LULL]
        weights = [probs[s] for s in states]
        new_state = self.rng.choice(states, p=weights)
        if new_state != self._state and self.verbose:
            print(f"  [BurstWrapper] step {self._step_count}: " f"{STATE_NAMES[self._state]} → {STATE_NAMES[new_state]}")
        self._state = new_state

    def _regime_features(self):
        feat = np.zeros(3, dtype=float)
        feat[self._state] = 1.0
        return feat

    def reset(self):
        self._state      = NORMAL
        self._step_count = 0
        base_obs = self.env.reset()
        return np.concatenate([base_obs, self._regime_features()])

    def seed(self, seed=None):
        return []

    def step(self, action):
        self._transition()
        self._step_count += 1

        base_obs, base_reward, done, info = self.env.step(action)

        shaped_reward = self._shape_reward(base_reward, info)

        obs = np.concatenate([base_obs, self._regime_features()])
        info['regime']       = self._state
        info['regime_name']  = STATE_NAMES[self._state]

        return obs, shaped_reward, done, info

    def _shape_reward(self, base_reward, info):
        violations = np.array(info['violations'], dtype=int)
        n_viol     = int(info['total_violations'])

        if n_viol == 0:
            eff_scale = REGIME_EFFICIENCY_SCALE[self._state]
            return base_reward * eff_scale

        pen_scale = REGIME_PENALTY_SCALE[self._state]
        
        shaped = -self.penalty * n_viol * pen_scale

        if self._state == BURST:
            urllc_viols = violations[self._urllc_start:self._urllc_end].sum()
            if urllc_viols > 0:
                shaped -= (self.penalty * urllc_viols * URLLC_BURST_EXTRA_PENALTY)

        return float(shaped)

    @property
    def n_prbs(self):
        return self.env.n_prbs

    @property
    def n_slices(self):
        return self.env.n_slices

    @property
    def n_variables(self):
        return self.env.n_variables + self.N_REGIME_FEATURES

    def current_regime(self):
        return STATE_NAMES[self._state]