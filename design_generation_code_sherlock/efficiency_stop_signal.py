#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import pickle
from nilearn.plotting import plot_design_matrix
#sys.path.append('/Users/jeanettemumford/Dropbox/Research/Projects/design_and_analysis/code_in_progress')
from utils import (calc_expected_run_num_by_chance, sample_shifted_truncated_exponential, 
                   est_eff_and_vif, est_psych_fitness)

args_in = sys.argv
filenum = args_in[1]

def make_stop_signal_timings(nstop, ngo, stop_dur, go_dur, blank_dur, nbreaks, 
                             break_fix_pre_message_dur, break_message_dur,  
                             break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                             isi_shift):
    '''
    Produces a randomly ordered set of trials for the stop signal task.
    Assumes a break between blocks of task.  Number of stop/go trials will be equal
    for each block of the task (nstop/(nbreaks + 1))
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A trial is structured as:
      fixation (based on truncated exponential) + 
      either a stop/go trial (duration = stop_dur/go_dur) +
      blank screen (duration = blank_dur) + 
      next trial (same structure as above until end of block is reached)
      At end of block the break begins with:
      fixation (length=break_dur) + 
      break message (break_message_dur) + 
      post message fixation (post_message_dur)
    Input:
      nstop: Total number of stop trials (will be evenly split between blocks)
             Integer divisible by (nbreaks + 1)
      ngo: Total number of go trials (will be evenly split between blocks)
           Integer divisible by (nbreaks + 1)
      stop_dur: Duration of a stop trial
      go_dur: Duration of a go trial
      blank_dur:  Duration of blank screen after stop/go stimulus offset
      nbreaks: Total number of breaks (blocks = nbreaks + 1)
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
       Includes stop/go/break message timings
    '''
    nblocks = nbreaks + 1
    ntrials = nstop + ngo
    ntrials_per_block = int(ntrials/nblocks)
    nstop_per_block = int(nstop / (nblocks))
    ngo_per_block = int(ngo / (nblocks))
    isi_vals, _ = sample_shifted_truncated_exponential(isi_exp_lam, isi_truncation, 
                                                    isi_shift, ntrials)

    # fragments = all sub-components of the run (go/stop/fixation/blank/break_message)
    # Each stimulus has 3 fragments and each break has 3 fragments
    isi_count = 0
    fragment_durations = []
    fragment_labels = []
    for cur_block in range(nblocks):
        stim_type_block = np.random.permutation(np.concatenate((np.repeat(['stop'], nstop_per_block),
                                                        np.repeat(['go'], ngo_per_block))))
        for cur_trial in range(ntrials_per_block):
            cur_stim_type = stim_type_block[cur_trial]
            stim_dur_cur_trial = stop_dur if cur_stim_type == 'stop' else go_dur
            fragment_durations.extend([isi_vals[isi_count], stim_dur_cur_trial, blank_dur])
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
    events_data = events_data.loc[(events_data.trial_type == 'stop') |
                    (events_data.trial_type == 'go') |
                    (events_data.trial_type == 'break_message')]
    return events_data


def run_stop_signal_eff_sim(nsim, nstop, ngo, stop_dur, go_dur, blank_dur, nbreaks, 
                                break_fix_pre_message_dur, break_message_dur,  
                                break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                                isi_shift, total_time, contrasts, avg_trial_repeats_info):
    '''
    Runs nsim randomly created stop signal designs through efficiency/vif/psych fitness
    measures and outputs results
    Input: 
      nsim:  Number of simulated design matrices
    Output:
      output: Pandas data frame with each fitness measure for each design
      events_all:  The events data for each design matrix
    '''
    output = {
        'eff_stop-go': [],
        'eff_stop': [],
        'eff_go': [],
        'eff_task': [],
        'vif_stop-go': [],
        'vif_stop': [],
        'vif_go': [],
        'vif_task': [],
        'scan_length':[],
        'kao_measure': [],
        'prob_runs_gte_2': [],
        'run_num_diff_from_avg': [],
        'prob_next_given_last1': [],
        'prob_next_given_last2': [],
        'pred_second_half_from_first': []
    }
    all_events = []

    for sim in range(nsim):
        events = make_stop_signal_timings(nstop, ngo, stop_dur, go_dur, blank_dur, nbreaks, 
                                break_fix_pre_message_dur, break_message_dur,  
                                break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                                isi_shift)
        if np.max(events.onset) > total_time:
            print('WARNING:  You need to increase the total time to fit all trials \n'
                'estimates from this simulation set should be discarded')
        eff_vals, vifs, _ = est_eff_and_vif(events, tr, total_time, contrasts)
        trials_no_breaks = events.trial_type[events.trial_type != 'break_message']
        # swap out stop/go for s/g (otherwise the est_psych_fitness function breaks)
        swaps = {'stop': 's', 'go': 'g'}
        trials_no_breaks = np.array(trials_no_breaks.replace(swaps))
        psych_assess = est_psych_fitness(trials_no_breaks, avg_trial_repeats_info)
        for key, val in eff_vals.items():
            output[f'eff_{key}'].append(val)
        for key, val in vifs.items():
            output[f'vif_{key}'].append(val)
        for key, val in psych_assess.items():
            output[key].append(val)
        all_events.append(events)
        output['scan_length'].append(
            events.onset.values[-1:][0] + events.duration.values[-1:][0] + 20)
    return pd.DataFrame(output), all_events


nstop = 60
ngo = 120
stop_dur = 1
go_dur = 1
blank_dur = .5
nbreaks = 2
break_fix_pre_message_dur = 6
break_message_dur = 4
break_fix_post_message_dur = 6
isi_exp_lam = 2
isi_truncation = 4.5
isi_shift = .5

tr = 1.49
total_time = 10*60
contrasts = {'stop': 'stop',
             'go': 'go',
             'stop-go': 'stop - go',
             'task': '.5*stop+.5*go'}



unpermuted_trials = np.concatenate([np.repeat('s', nstop/3), 
                                    np.repeat('g', ngo/3)],
                                    axis=0)
avg_trial_repeats_info = calc_expected_run_num_by_chance(unpermuted_trials, nsims=5000)
avg_trial_repeats_info['g_run_counts'] = avg_trial_repeats_info['g_run_counts'] * 3
avg_trial_repeats_info['s_run_counts'] = avg_trial_repeats_info['s_run_counts'] * 3



nsim = 20000 # Should take 1 hour

output_setting1, events_setting1 = run_stop_signal_eff_sim(nsim, nstop, ngo, stop_dur, go_dur, blank_dur, nbreaks, 
                                break_fix_pre_message_dur, break_message_dur,  
                                break_fix_post_message_dur, isi_exp_lam, isi_truncation, 
                                isi_shift, total_time, contrasts, avg_trial_repeats_info)



output_setting1.to_csv(f'/home/users/jmumford/efficiency/output/output_{filenum}.csv')

with open(f'/home/users/jmumford/efficiency/output/events_{filenum}.pkl', 'wb') as f: 
    pickle.dump(events_setting1, f)

#with open(f'/home/users/jmumford/efficiency/output/events_22.pkl', "rb") as f: 
#    # "rb" because we want to read in binary mode
#    events = pickle.load(f)    