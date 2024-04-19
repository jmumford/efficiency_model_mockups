#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utils import (calc_expected_run_num_by_chance, sample_shifted_truncated_exponential, 
                   est_eff_and_vif, run_eff_sim, calc_avg_prob_next_given_last1_and_last2)



def make_nback_timings(nblocks, n_act_trials_per_block, stim_dur, blank_dur,  
                             break_fix_pre_message_dur, break_message_dur,  
                             break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                             isi_shift):
    '''
    Set up the nback task.  Blocks will alternate between 1-back and 2-back with breaks
    between each task block
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A  block of trials repeats the following n_trials_per_block times
      but 1 extra trial is added before a 1-back block and 2 extra trials before a 2-back block
      trial is structured as:
      fixation (based on truncated exponential) + 
      stimulus (stim_dur) + blank (blank_dur) + 
      next trial (same structure as above until end of block is reached)
      At end of block the break begins with:
      fixation (length=break_dur) + 
      break message (break_message_dur) + 
      post message fixation (post_message_dur)
    Input:
      n_block: Total number of blocks (will be evenly split between 1-back and 2-back)
      cue_stim_dur/blank_dur: Durations of cue_stim/blank
      blank_dur:  Duration of blank screen after stimulus offset
      break_fix_pre_message_dur: Fixation duration at beginning of break (no jittering)
      break_message_dur:  Duration of message shown during break
      break_fix_post_message_dur:  Fixation duration after message (no jittering)
      isi_exp_lam:  (seconds) The lambda parameter of the exponential used for the isi 
                    (inverse is the non-truncated/non-shifted mean)
      isi_truncation: (seconds) Truncation value of the exponential *NOT* including shift.  
                      Max ISI = isi_truncation + isi_shift
      isi_shift: (seconds) Shift value for isi (minimum isi value)
    Output:
       Events pandas data frame with onsets, trial_type and duration.  
    '''
    if nblocks % 2 != 0:
        raise ValueError('nblocks must be even')
    n_each_back = int(nblocks/2)
    #add in burn in trials
    ntrials = int(n_act_trials_per_block * nblocks + (3*n_each_back))
    block_type = ['one_back','two_back']*n_each_back
    
    fragment_durations = []
    fragment_labels = []
    for block in block_type:
        if block == 'one_back':
            n_extra = 1
        else:
            n_extra = 2
        ntrials_block = n_act_trials_per_block + n_extra
        isi_vals_block, _ = sample_shifted_truncated_exponential(isi_exp_lam, isi_truncation, 
                                          isi_shift, ntrials_block)
        fragment_durations_block = []
        for isi_val in isi_vals_block:
            fragment_durations_block.extend([isi_val, stim_dur, blank_dur])
        fragment_durations_block.extend([break_fix_pre_message_dur, break_message_dur, break_fix_post_message_dur])
        fragment_labels_block = ['fixation', 'starter_trial', 'break']*n_extra + \
                                ['fixation', block, 'break'] * n_act_trials_per_block + \
                                ['fix_break', 'break_message', 'fix_break']
        fragment_durations.extend(fragment_durations_block)
        fragment_labels.extend(fragment_labels_block)

    # If you don't want the beginning of the run to start at 0, change this 
    # (e.g. if you want to add the 10s to reach steady state)
    run_start = 0
    fragment_onsets = np.cumsum([run_start] + fragment_durations)[:-1]       
    events_data = pd.DataFrame({'onset': fragment_onsets,
                                'trial_type': fragment_labels,
                                'duration': fragment_durations})
    events_data = events_data.loc[events_data['trial_type'].str.contains('starter|back|break_message')==True]
    return events_data


if __name__ == "__main__":
    args_in = sys.argv
    filenum = args_in[1]


    events_inputs = {
        'nblocks': 10,
        'n_act_trials_per_block': 12,
        'stim_dur': 1,
        'blank_dur': .5,
        'break_fix_pre_message_dur': .5,
        'break_message_dur': 4,
        'break_fix_post_message_dur': .5,
        'isi_exp_lam': 2,
        'isi_truncation': 4.5,
        'isi_shift': .5
    }

    tr = 1.49
    total_time = 7*60
    contrasts = {'1back_v_baseline': 'one_back',
                '2back_v_baseline': 'two_back',
                'all_task': '.5*one_back + .5*two_back',
                '2back-1back': 'two_back-one_back' 
                }

    trials_psych_assess_map = {'one_back': '1', 'two_back': '2'}

    ntrials = events_inputs['n_act_trials_per_block'] * events_inputs['nblocks']
    unpermuted_trials = np.concatenate([np.repeat('1', ntrials/2), 
                                        np.repeat('2', ntrials/2)],
                                        axis=0)
    avg_trial_repeats_info = calc_expected_run_num_by_chance(unpermuted_trials, nsims=5000)
    avg_prob_given_last1, avg_prob_given_last2 = calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)

    nsim = 20000
    output_setting1, events_setting1 = run_eff_sim(nsim, events_inputs, make_nback_timings, 
                                contrasts, avg_trial_repeats_info, tr, total_time,
                                trials_psych_assess_map, avg_prob_given_last1, avg_prob_given_last2)
    output_setting1.to_csv(f'/home/users/jmumford/efficiency/output/nback_output_{filenum}.csv')
    with open(f'/home/users/jmumford/efficiency/output/nback_events_{filenum}.pkl', 'wb') as f: 
        pickle.dump(events_setting1, f)
    