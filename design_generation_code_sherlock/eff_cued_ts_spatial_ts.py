#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utils import (calc_expected_run_num_by_chance, sample_shifted_truncated_exponential, 
                   est_eff_and_vif, run_eff_sim, calc_avg_prob_next_given_last1_and_last2)


def make_cued_spatial_ts_timings(n_tsw_csw, n_tst_cst, n_tst_csw, cue_stim_dur, blank_dur,  
                             nbreaks, break_fix_pre_message_dur, break_message_dur,  
                             break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                             isi_shift):
    '''
    Produces a randomly ordered set of trials for the cued and spatial ts tasks.
    Assumes a break between blocks of task.  Number of tsw_csw/tst_cst/tst_csw trials will be equal
    for each block of the task 
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A trial is structured as:
      fixation (based on truncated exponential) + 
      cue_stim (cue_stim_dur) + blank (blank_dur) + 
      next trial (same structure as above until end of block is reached)
      At end of block the break begins with:
      fixation (length=break_dur) + 
      break message (break_message_dur) + 
      post message fixation (post_message_dur)
    Input:
      n_tsw_csw, n_tst_cst, n_tst_csw: Total number of tsw_csw/tst_cst/tst_csw trials 
          (will be evenly split between blocks) Integer divisible by (nbreaks + 1)
      cue_stim_dur/blank_dur: Durations of cue_stim/blank
      blank_dur:  Duration of blank screen after stop/go stimulus offset
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
    
    nblocks = nbreaks + 1
    ntrials = n_tsw_csw + n_tst_cst + n_tst_csw + 3 # add 3 for the n/a trials
    ntrials_per_block = int(ntrials/nblocks)
    n_tsw_csw_per_block = int(n_tsw_csw / (nblocks))
    n_tst_cst_per_block = int(n_tst_cst / (nblocks))
    n_tst_csw_per_block = int(n_tst_csw / (nblocks))
    isi_vals, _ = sample_shifted_truncated_exponential(isi_exp_lam, isi_truncation, 
                                                    isi_shift, ntrials)

    # fragments = all sub-components of the run (go/stop/fixation/blank/break_message)
    # Each stimulus has 3 fragments and each break has 3 fragments
    isi_count = 0
    fragment_durations = []
    fragment_labels = []
    for cur_block in range(nblocks):
        stim_type_block = np.random.permutation(np.concatenate((np.repeat(['tsw_csw'], n_tsw_csw_per_block),
                                                        np.repeat(['tst_cst'], n_tst_cst_per_block),
                                                        np.repeat(['tst_csw'], n_tst_csw_per_block))))
        # add in starter trial (n/a trial)
        stim_type_block = np.insert(stim_type_block,0, ['start'])
        for cur_trial in range(ntrials_per_block):
            cur_stim_type = stim_type_block[cur_trial]
            fragment_durations.extend([isi_vals[isi_count], cue_stim_dur, blank_dur])
            fragment_labels.extend(['isi_fix', cur_stim_type, 'blank'])
            isi_count = isi_count + 1
        #Add break
        fragment_durations.extend([break_fix_pre_message_dur, break_message_dur, break_fix_post_message_dur])
        fragment_labels.extend(['fix_break', 'break_message', 'fix_break'])

    # If you don't want the beginning of the run to start at 0, change this 
    # (e.g. if you want to add the 10s to reach steady state)
    run_start = 0
    fragment_onsets = np.cumsum([run_start] + fragment_durations)[:-1]       
    events_data = pd.DataFrame({'onset': fragment_onsets,
                                'trial_type': fragment_labels,
                                'duration': fragment_durations})
    events_data = events_data.loc[events_data['trial_type'].str.contains('start|tsw_csw|tst_cst|tst_csw|break_message')==True]
    return events_data


if __name__ == "__main__":
    args_in = sys.argv
    filenum = args_in[1]


    events_inputs = {
        'n_tsw_csw': 96,
        'n_tst_cst': 48,
        'n_tst_csw': 48,
        'nbreaks': 2,
        'cue_stim_dur': 1.15,
        'blank_dur': .5,
        'break_fix_pre_message_dur': 6,
        'break_message_dur': 4,
        'break_fix_post_message_dur': 6,
        'isi_exp_lam': 2,
        'isi_truncation': 4.5,
        'isi_shift': .5
    }

    tr = 1.49
    total_time = 12*60
    contrasts = {'tsw_csw-tst_csw': 'tsw_csw-tst_csw',
                'tst_csw-tst_cst': 'tst_csw - tst_cst',
                'all_task': '.333*tsw_csw + .333*tst_csw + .333*tst_cst' 
             }

    trials_psych_assess_map = {'tsw_csw': '1', 'tst_csw': '2', 'tst_cst': '3'}

    unpermuted_trials = np.concatenate([np.repeat('1', events_inputs['n_tsw_csw']/3), 
                                        np.repeat('2', events_inputs['n_tst_csw']/3),
                                        np.repeat('3', events_inputs['n_tst_cst']/3)],
                                        axis=0)
    avg_trial_repeats_info = calc_expected_run_num_by_chance(unpermuted_trials, nsims=5000)

    for val in ['1', '2', '3']:
        avg_trial_repeats_info[f'{val}_run_counts'] = avg_trial_repeats_info[f'{val}_run_counts'] * 3
    avg_prob_given_last1, avg_prob_given_last2 = calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)
    nsim = 20000
    output_setting1, events_setting1 = run_eff_sim(nsim, events_inputs, make_cued_spatial_ts_timings, 
                                contrasts, avg_trial_repeats_info, tr, total_time,
                                trials_psych_assess_map, avg_prob_given_last1, avg_prob_given_last2)

    output_setting1.to_csv(f'/home/users/jmumford/efficiency/output/cued_ts_spatial_ts_output_{filenum}.csv')
    with open(f'/home/users/jmumford/efficiency/output/cued_ts_spatial_ts_events_{filenum}.pkl', 'wb') as f: 
        pickle.dump(events_setting1, f)


