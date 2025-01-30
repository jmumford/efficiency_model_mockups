#!/usr/bin/env python3

import pickle
import sys

import numpy as np
import pandas as pd

from utils import (
    calc_avg_prob_next_given_last1_and_last2,
    calc_expected_run_num_by_chance,
    run_eff_sim,
    sample_shifted_truncated_exponential,
)


def make_stop_flank_stroop_gng_timings(
    nc1,
    nc2,
    c1_dur,
    c2_dur,
    blank_dur,
    nbreaks,
    break_fix_pre_message_dur,
    break_message_dur,
    break_fix_post_message_dur,
    isi_exp_lam,
    isi_truncation,
    isi_shift,
):
    """
    Produces a randomly ordered set of trials for the stop signal task, flanker, strop and go/nogo.
    Assumes a break between blocks of task.  Number of stop/go trials will be equal
    for each block of the task (nc1/(nbreaks + 1))
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A trial is structured as:
      fixation (based on truncated exponential) +
      either a stop/go trial (duration = c1_dur/c2_dur) +
      blank screen (duration = blank_dur) +
      next trial (same structure as above until end of block is reached)
      At end of block the break begins with:
      fixation (length=break_dur) +
      break message (break_message_dur) +
      post message fixation (post_message_dur)
    Input:
      nc1: Total number of condition 1 (will be evenly split between blocks)
             Integer divisible by (nbreaks + 1)
      nc2: Total number of condition 2 (will be evenly split between blocks)
           Integer divisible by (nbreaks + 1)
      c1_dur: Duration of a stop trial
      c2_dur: Duration of a go trial
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
    """
    nblocks = nbreaks + 1
    ntrials = nc1 + nc2
    ntrials_per_block = int(ntrials / nblocks)
    nc1_per_block = int(nc1 / (nblocks))
    nc2_per_block = int(nc2 / (nblocks))
    isi_vals, _ = sample_shifted_truncated_exponential(
        isi_exp_lam, isi_truncation, isi_shift, ntrials
    )

    # fragments = all sub-components of the run (go/stop/fixation/blank/break_message)
    # Each stimulus has 3 fragments and each break has 3 fragments
    isi_count = 0
    fragment_durations = []
    fragment_labels = []
    for cur_block in range(nblocks):
        stim_type_block = np.random.permutation(
            np.concatenate(
                (
                    np.repeat(['cond1'], nc1_per_block),
                    np.repeat(['cond2'], nc2_per_block),
                )
            )
        )
        for cur_trial in range(ntrials_per_block):
            cur_stim_type = stim_type_block[cur_trial]
            stim_dur_cur_trial = c1_dur if cur_stim_type == 'cond1' else c2_dur
            fragment_durations.extend(
                [isi_vals[isi_count], stim_dur_cur_trial, blank_dur]
            )
            fragment_labels.extend(['isi_fix', cur_stim_type, 'blank'])
            isi_count = isi_count + 1
        # Add break
        fragment_durations.extend(
            [break_fix_pre_message_dur, break_message_dur, break_fix_post_message_dur]
        )
        fragment_labels.extend(['fix_break', 'break_message', 'fix_break'])

    # If you don't want the beginning of the run to start at 0, change this
    # (e.g. if you want to add the 10s to reach steady state)
    run_start = 0
    fragment_onsets = np.cumsum([run_start] + fragment_durations)[:-1]
    events_data = pd.DataFrame(
        {
            'onset': fragment_onsets,
            'trial_type': fragment_labels,
            'duration': fragment_durations,
        }
    )
    events_data = events_data.loc[
        events_data['trial_type'].str.contains('cond1|cond2|break_message') == True
    ]
    return events_data


if __name__ == '__main__':
    args_in = sys.argv
    filenum = args_in[1]
    task = args_in[2]

    if task == 'stop':
        events_inputs = {
            'nc1': 60,
            'nc2': 120,
            'c1_dur': 1,
            'c2_dur': 1,
            'blank_dur': 0.5,
            'nbreaks': 2,
            'break_fix_pre_message_dur': 6,
            'break_message_dur': 4,
            'break_fix_post_message_dur': 6,
            'isi_exp_lam': 2,
            'isi_truncation': 4.5,
            'isi_shift': 0.5,
        }
        tr = 1.49
        total_time = 10 * 60
        contrasts = {
            'stop': 'cond1',
            'go': 'cond2',
            'stop-go': 'cond1 - cond2',
            'task': '.5*cond1+.5*cond2',
        }
        trials_psych_assess_map = {'cond1': '1', 'cond2': '2'}
        name_swap = {'cond1': 'stop', 'cond2': 'go'}
        unpermuted_trials = np.concatenate(
            [
                np.repeat('1', events_inputs['nc1'] / 3),
                np.repeat('2', events_inputs['nc2'] / 3),
            ],
            axis=0,
        )
        avg_trial_repeats_info = calc_expected_run_num_by_chance(
            unpermuted_trials, nsims=5000
        )
        for val in ['1', '2']:
            avg_trial_repeats_info[f'{val}_run_counts'] = (
                avg_trial_repeats_info[f'{val}_run_counts'] * 3
            )
        avg_prob_given_last1, avg_prob_given_last2 = (
            calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)
        )
        nsim = 20000
        output_setting1, events_setting1 = run_eff_sim(
            nsim,
            events_inputs,
            make_stop_flank_stroop_gng_timings,
            contrasts,
            avg_trial_repeats_info,
            tr,
            total_time,
            trials_psych_assess_map,
            avg_prob_given_last1,
            avg_prob_given_last2,
            deriv=False,
            name_swap=name_swap,
        )
        output_setting1.to_csv(
            f'/home/users/jmumford/efficiency/output/stop_signal_output_{filenum}.csv'
        )
        with open(
            f'/home/users/jmumford/efficiency/output/stop_signal_events_{filenum}.pkl',
            'wb',
        ) as f:
            pickle.dump(events_setting1, f)

    if task == 'flanker':
        events_inputs_flanker = {
            'nc1': 60,
            'nc2': 60,
            'c1_dur': 1,
            'c2_dur': 1,
            'blank_dur': 0.5,
            'nbreaks': 2,
            'break_fix_pre_message_dur': 6,
            'break_message_dur': 4,
            'break_fix_post_message_dur': 6,
            'isi_exp_lam': 2,
            'isi_truncation': 4.5,
            'isi_shift': 0.5,
        }

        tr = 1.49
        total_time = 6 * 60
        contrasts_flanker = {
            'congruent': 'cond1',
            'incongruent': 'cond2',
            'incongruent-congruent': 'cond1 - cond2',
            'task': '.5*cond1+.5*cond2',
        }
        trials_psych_assess_map = {'cond1': '1', 'cond2': '2'}
        name_swap = {'cond1': 'congruent', 'cond2': 'incongruent'}
        unpermuted_trials = np.concatenate(
            [
                np.repeat('1', events_inputs_flanker['nc1'] / 3),
                np.repeat('2', events_inputs_flanker['nc2'] / 3),
            ],
            axis=0,
        )
        avg_trial_repeats_info = calc_expected_run_num_by_chance(
            unpermuted_trials, nsims=5000
        )
        for val in ['1', '2']:
            avg_trial_repeats_info[f'{val}_run_counts'] = (
                avg_trial_repeats_info[f'{val}_run_counts'] * 3
            )
        avg_prob_given_last1, avg_prob_given_last2 = (
            calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)
        )
        nsim = 20000
        output_setting1, events_setting1 = run_eff_sim(
            nsim,
            events_inputs_flanker,
            make_stop_flank_stroop_gng_timings,
            contrasts_flanker,
            avg_trial_repeats_info,
            tr,
            total_time,
            trials_psych_assess_map,
            avg_prob_given_last1,
            avg_prob_given_last2,
            deriv=False,
            name_swap=name_swap,
        )
        output_setting1.to_csv(
            f'/home/users/jmumford/efficiency/output/flanker_stroop_output_{filenum}.csv'
        )
        with open(
            f'/home/users/jmumford/efficiency/output/flanker_stroop_events_{filenum}.pkl',
            'wb',
        ) as f:
            pickle.dump(events_setting1, f)

    if task == 'gng':
        events_inputs_gng = {
            'nc1': 162,
            'nc2': 27,
            'c1_dur': 1,
            'c2_dur': 1,
            'blank_dur': 0.5,
            'nbreaks': 2,
            'break_fix_pre_message_dur': 6,
            'break_message_dur': 4,
            'break_fix_post_message_dur': 6,
            'isi_exp_lam': 2,
            'isi_truncation': 4.5,
            'isi_shift': 0.5,
        }
        tr = 1.49
        total_time = 10 * 60
        contrasts_gng = {
            'go': 'cond1',
            'nogo': 'cond2',
            'nogo-go': 'cond2 - cond1',
            'task': '.5*cond1+.5*cond2',
        }
        trials_psych_assess_map = {'cond1': '1', 'cond2': '2'}
        name_swap = {'cond1': 'go', 'cond2': 'nogo'}
        unpermuted_trials = np.concatenate(
            [
                np.repeat('1', events_inputs_gng['nc1'] / 3),
                np.repeat('2', events_inputs_gng['nc2'] / 3),
            ],
            axis=0,
        )
        avg_trial_repeats_info = calc_expected_run_num_by_chance(
            unpermuted_trials, nsims=5000
        )

        for val in ['1', '2']:
            avg_trial_repeats_info[f'{val}_run_counts'] = (
                avg_trial_repeats_info[f'{val}_run_counts'] * 3
            )
        avg_prob_given_last1, avg_prob_given_last2 = (
            calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)
        )
        nsim = 20000
        output_setting1, events_setting1 = run_eff_sim(
            nsim,
            events_inputs_gng,
            make_stop_flank_stroop_gng_timings,
            contrasts_gng,
            avg_trial_repeats_info,
            tr,
            total_time,
            trials_psych_assess_map,
            avg_prob_given_last1,
            avg_prob_given_last2,
            deriv=False,
            name_swap=name_swap,
        )
        output_setting1.to_csv(
            f'/home/users/jmumford/efficiency/output/gng_output_{filenum}.csv'
        )
        with open(
            f'/home/users/jmumford/efficiency/output/gng_events_{filenum}.pkl', 'wb'
        ) as f:
            pickle.dump(events_setting1, f)
