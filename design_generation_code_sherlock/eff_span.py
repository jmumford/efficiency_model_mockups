#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utils import (calc_expected_run_num_by_chance, sample_shifted_truncated_exponential, 
                   est_eff_and_vif, run_eff_sim, calc_avg_prob_next_given_last1_and_last2)


def make_span_timings(ntrials, proc_dur, store_dur, recall_dur,  
                             nbreaks, break_fix_pre_message_dur, break_message_dur,  
                             break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                             isi_shift):
    '''
    Generates events timings for Span task assuming a single condition is presented for the entire run
    Assumes a break between blocks of task.  
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A trial is structured as:
      EMC period = 4 repeats of [symm(or fix) (proc_dur seconds) + TBR(or fix) (store_dur seconds)] +
      GRID/recall period (5s) +
      fixation (min=2s, max=20s, mean=5s)
      At end of block the break begins with:
      fixation (length=break_dur) + 
      break message (break_message_dur) + 
      post message fixation (post_message_dur)
    Input:
      n_trials: Total number of trials.  
      proc_dur: Processing durations (sym or fixation)
      store_dur: TBR or fixation duration 
      break_fix_pre_message_dur: Fixation duration at beginning of break (no jittering)
      break_message_dur:  Duration of message shown during break
      break_fix_post_message_dur:  Fixation duration after message (no jittering)
      isi_exp_lam:  (seconds) The lambda parameter of the exponential used for the isi 
                    (inverse is the non-truncated/non-shifted mean)
      isi_truncation: (seconds) Truncation value of the exponential *NOT* including shift.  
                      Max ISI = isi_truncation + isi_shift
      isi_shift: (seconds) Shift value for isi (minimum isi value)
      model_option: One of ['one_component', 'two_components', 'three_components'].  Refers to how cue
                    cti and stimulus are modeled.  Three components models them separately, two models cue+cti together,
                    and one component models cue+cti+stimulus as a single regressor.
    Output:
       Events pandas data frame with onsets, trial_type and duration.  
    '''
    
    nblocks = nbreaks + 1
    ntrials_per_block = int(ntrials/nblocks)
    
    isi_vals, _ = sample_shifted_truncated_exponential(isi_exp_lam, isi_truncation, 
                                                    isi_shift, ntrials)

    # Each stimulus has 3 fragments and each break has 3 fragments
    isi_count = 0
    fragment_durations = []
    fragment_labels = []
    for cur_block in range(nblocks):
        for cur_trial in range(ntrials_per_block):
            fragment_durations.extend([isi_vals[isi_count]] + 4*[proc_dur, store_dur] + [recall_dur])
            fragment_labels.extend(['isi_fix'] + [f'first_proc_period', f'_first_store_period'] + 3*[f'proc_period', f'store_period'] + [f'recall'])
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
    events_data = events_data.loc[events_data['trial_type'].str.contains('first_proc|recall|break_message')==True]
    events_data.loc[events_data['trial_type'].str.contains('proc'), 'duration'] = 4*(proc_dur + store_dur)
    events_data.trial_type = events_data.trial_type.str.replace('first_proc_period', 'emc')
    return events_data


if __name__ == "__main__":
    args_in = sys.argv
    filenum = args_in[1]

    events_inputs = {
        'ntrials': 24,
        'proc_dur' : 3, 
        'store_dur': 1, 
        'recall_dur': 5,
        'nbreaks': 2,
        'break_fix_pre_message_dur': 6,
        'break_message_dur': 4,
        'break_fix_post_message_dur': 6,
        'isi_exp_lam': .33,
        'isi_truncation': 20,
        'isi_shift': 2
    }

    tr = 1.49
    total_time =  60*12

    trials_psych_assess_map = {'proc_period': '1'}

    unpermuted_trials = np.repeat('1', events_inputs['ntrials'])
    avg_trial_repeats_info = calc_expected_run_num_by_chance(unpermuted_trials, nsims=5000)

    for val in ['1']:
        avg_trial_repeats_info[f'{val}_run_counts'] = avg_trial_repeats_info[f'{val}_run_counts'] * 3
    avg_prob_given_last1, avg_prob_given_last2 = calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)

    contrasts = {
        'recall vs baseline': 'recall',
        'emc vs baseline': 'emc'
    }

    nsim = 20000
    output_setting1, events_setting1 = run_eff_sim(nsim, events_inputs, make_span_timings, 
                                contrasts, avg_trial_repeats_info, tr, total_time,
                                trials_psych_assess_map, avg_prob_given_last1, avg_prob_given_last2, 
                                deriv=False, est_psych=False)


    output_setting1.to_csv(f'/home/users/jmumford/efficiency/output/span_output_{filenum}.csv')
    with open(f'/home/users/jmumford/efficiency/output/span_events_{filenum}.pkl', 'wb') as f: 
        pickle.dump(events_setting1, f)
