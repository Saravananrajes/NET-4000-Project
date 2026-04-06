import sys
sys.path.insert(0, '.')
sys.path.insert(0, './gym-ran_slice')

import gym
from itertools import count
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC
from slice_ran import SliceRANmMTC, SliceRANeMBB
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset
from numpy.random import default_rng as np_default_rng

ext_scenario_1 = {
    'n_prbs':  200,
    'n_embb':  3,
    'n_urllc': 2,
    'n_mmtc':  0,
}

ext_scenario_2 = {
    'n_prbs':  200,
    'n_embb':  2,
    'n_urllc': 2,
    'n_mmtc':  2,
}

ext_scenario_3 = {
    'n_prbs':  150,
    'n_embb':  1,
    'n_urllc': 2,
    'n_mmtc':  4,
}

ext_scenario_4 = {
    'n_prbs':  150,
    'n_embb':  2,
    'n_urllc': 1,
    'n_mmtc':  2,
}

ext_scenarios = [ext_scenario_1, ext_scenario_2, ext_scenario_3, ext_scenario_4]

CBR_description_embb = {
    'lambda':   2.0 / 60.0,
    't_mean':   30.0,
    'bit_rate': 500000,
}

VBR_description_embb = {
    'lambda': 5.0 / 60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1,
}

SLA_embb = {
    'cbr_th':    10e6,
    'cbr_prb':   20,
    'cbr_queue': 10e4,
    'vbr_th':    15e6,
    'vbr_prb':   30,
    'vbr_queue': 15e4,
}

CBR_description_urllc = {
    'lambda':   4.0 / 60.0,   
    't_mean':   10.0,         
    'bit_rate': 1000000,      
}

VBR_description_urllc = {
    'lambda': 8.0 / 60.0,    
    't_mean': 10.0,
    'p_size': 500,             
    'b_size': 200,
    'b_rate': 1,
}

SLA_urllc = {
    'cbr_th':    20e6,      
    'cbr_prb':   15,        
    'cbr_queue': 1e3,         
    'vbr_th':    25e6,      
    'vbr_prb':   20,
    'vbr_queue': 2e3,          
}

MTC_description = {
    'n_devices':      1000,
    'repetition_set': [2, 4, 8, 16, 32, 64, 128],
    'period_set':     [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000],
}

SLA_mmtc = {
    'delay': 300,
}

state_variables_embb = [
    'cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr',
    'vbr_traffic', 'vbr_th', 'vbr_prb', 'vbr_queue', 'vbr_snr',
]

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

def create_env(rng, n, slots_per_step=50,
                   propagation_type='macro_cell_urban_2GHz',
                   penalty=1000):
    """
    Create an extended RAN slicing environment with eMBB + URLLC + mMTC slices.

    Parameters
    ----------
    rng            : numpy default_rng instance
    n              : scenario index (0-3)
    slots_per_step : radio slots per decision step (default 50 = 50ms)
    penalty        : SLA violation penalty coefficient
    """
    sc        = ext_scenarios[n]
    n_prbs    = sc['n_prbs']
    n_embb    = sc['n_embb']
    n_urllc   = sc['n_urllc']
    n_mmtc    = sc['n_mmtc']
    time_step = slots_per_step * 1e-3

    norm_embb = {
        'cbr_traffic': 5e6  * time_step,
        'cbr_th':      10e6 * time_step,
        'cbr_prb':     25   * slots_per_step,
        'cbr_queue':   10e4 * slots_per_step,
        'cbr_snr':     35   * slots_per_step,
        'vbr_traffic': 5e6  * time_step,
        'vbr_th':      10e6 * time_step,
        'vbr_prb':     35   * slots_per_step,
        'vbr_queue':   10e4 * slots_per_step,
        'vbr_snr':     35   * slots_per_step,
    }

    norm_urllc = {
        'cbr_traffic': 10e6 * time_step,
        'cbr_th':      20e6 * time_step,
        'cbr_prb':     20   * slots_per_step,
        'cbr_queue':   1e3  * slots_per_step,
        'cbr_snr':     35   * slots_per_step,
        'vbr_traffic': 10e6 * time_step,
        'vbr_th':      25e6 * time_step,
        'vbr_prb':     25   * slots_per_step,
        'vbr_queue':   2e3  * slots_per_step,
        'vbr_snr':     35   * slots_per_step,
    }

    norm_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay':   100 * slots_per_step,
    }

    def new_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb,
                            CBR_description_embb, VBR_description_embb,
                            state_variables_embb, norm_embb, slots_per_step)

    def new_urllc(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_urllc,
                            CBR_description_urllc, VBR_description_urllc,
                            state_variables_embb, norm_urllc, slots_per_step)

    def new_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description,
                            state_variables_mmtc, norm_mmtc, slots_per_step)

    mcs_codeset  = MCSCodeset()
    user_counter = count()
    slices_l1    = []

    for id in range(n_embb):
        slice_rng = np_default_rng(rng.integers(1_000_000_000))
        snr_gen   = SINRSelectiveFading(slice_rng, propagation_type, n_prbs=n_prbs)
        scheduler = ProportionalFair(mcs_codeset)
        sl = SliceL1eMBB(slice_rng, snr_gen, 20,
                        [new_embb(id, slice_rng, user_counter)], scheduler)
        slices_l1.append(sl)

    for id in range(n_urllc):
        slice_rng = np_default_rng(rng.integers(1_000_000_000))
        snr_gen   = SINRSelectiveFading(slice_rng, propagation_type, n_prbs=n_prbs)
        scheduler = ProportionalFair(mcs_codeset)
        sl = SliceL1eMBB(slice_rng, snr_gen, 20,
                        [new_urllc(id, slice_rng, user_counter)], scheduler)
        slices_l1.append(sl)

    for id in range(n_mmtc):
        sl = SliceL1mMTC(5, [new_mmtc(id, rng)])
        slices_l1.append(sl)

    node     = NodeB(slices_l1, slots_per_step, n_prbs)
    node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b=node, penalty=penalty)

    return node_env, n_embb, n_urllc, n_mmtc


def get_slice_layout(n):
    """
    Returns (n_embb, n_urllc, n_mmtc) for scenario n.
    Slice order in obs/action: [eMBB...] [URLLC...] [mMTC...]
    """
    sc = ext_scenarios[n]
    return sc['n_embb'], sc['n_urllc'], sc['n_mmtc']