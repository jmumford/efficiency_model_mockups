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


def make_spatial_cueing_timings(
    n_no_cue,
    n_double_cue,
    n_valid_cue,
    n_invalid_cue,
    cue_dur,
    cti_durs,
    stim_dur,
    blank_dur,
    nbreaks,
    break_fix_pre_message_dur,
    break_message_dur,
    break_fix_post_message_dur,
    isi_exp_lam,
    isi_truncation,
    isi_shift,
    model_option,
):
    """
    Produces a randomly ordered set of trials for the spatial cueing task.
    Assumes a break between blocks of task.  Number of no_cue/double_cue/valid_cue trials will be equal
    for each block of the task
    ISI is sampled from a truncated exponential.  Note the truncation value for the
    truncated exponential does *not* include the shift, so the max isi is
    isi_truncation + isi_shift.
    A trial is structured as:
      fixation (based on truncated exponential) +
      cue(cue_dur) + CTI (3 values for duration) + stimulus (stim_dur) + blank (blank_dur) +
      next trial (same structure as above until end of block is reached)
      At end of block the break begins with:
      fixation (length=break_dur) +
      break message (break_message_dur) +
      post message fixation (post_message_dur)
    Input:
      n_no_cue, n_double_cue, n_valid_cue: Total number of no_cue/double_cue/valid_cue trials
          (will be evenly split between blocks) Integer divisible by (nbreaks + 1)
      cue_dur/cti_dur/stim_dur/blank_dur: Durations of cue/cti/stim/blank.
                                          All are integers, but cti, which is a vector of 3 values
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
    """

    nblocks = nbreaks + 1
    ntrials = n_no_cue + n_double_cue + n_valid_cue + n_invalid_cue
    ntrials_per_block = int(ntrials / nblocks)
    n_no_cue_per_block = int(n_no_cue / (nblocks))
    n_double_cue_per_block = int(n_double_cue / (nblocks))
    n_valid_cue_per_block = int(n_valid_cue / (nblocks))
    n_invalid_cue_per_block = int(n_invalid_cue / (nblocks))
    isi_vals, _ = sample_shifted_truncated_exponential(
        isi_exp_lam, isi_truncation, isi_shift, ntrials
    )

    # Each stimulus has 3 fragments and each break has 3 fragments
    isi_count = 0
    fragment_durations = []
    fragment_labels = []
    for cur_block in range(nblocks):
        cue_types = ['no_cue', 'double_cue', 'valid_cue', 'invalid_cue']
        cue_nums = [
            n_no_cue_per_block,
            n_double_cue_per_block,
            n_valid_cue_per_block,
            n_invalid_cue_per_block,
        ]
        stim_type_cti_val = []
        num_cti_vals = len(cti_durs)
        for cue_type, cue_num in zip(cue_types, cue_nums):
            for cti_dur in cti_durs:
                stim_type_cti_val.extend(
                    [[cue_type, cti_dur]] * int(cue_num / num_cti_vals)
                )
        stim_type_cti_val_perm = np.random.permutation(np.array(stim_type_cti_val))
        for cur_trial in range(ntrials_per_block):
            cur_stim_type = stim_type_cti_val_perm[cur_trial, 0]
            cur_cti_dur_s = float(stim_type_cti_val_perm[cur_trial, 1])
            if model_option == 'three_components':
                fragment_durations.extend(
                    [isi_vals[isi_count], cue_dur, cur_cti_dur_s, stim_dur, blank_dur]
                )
                fragment_labels.extend(
                    [
                        'isi_fix',
                        f'cue_{cur_stim_type}',
                        'cti',
                        f'stim_{cur_stim_type}_{int(1000*cur_cti_dur_s)}',
                        'blank',
                    ]
                )
            if model_option == 'two_components':
                fragment_durations.extend(
                    [isi_vals[isi_count], cue_dur + cur_cti_dur_s, stim_dur, blank_dur]
                )
                fragment_labels.extend(
                    [
                        'isi_fix',
                        f'cue_{cur_stim_type}_cti',
                        f'stim_{cur_stim_type}_{int(1000*cur_cti_dur_s)}',
                        'blank',
                    ]
                )
            if model_option == 'one_component':
                fragment_durations.extend(
                    [isi_vals[isi_count], cue_dur + cur_cti_dur_s + stim_dur, blank_dur]
                )
                fragment_labels.extend(
                    [
                        'isi_fix',
                        f'cue_cti_stim_{cur_stim_type}_{int(1000*cur_cti_dur_s)}',
                        'blank',
                    ]
                )
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
        events_data['trial_type'].str.contains('cue|cti|stim|break_message') == True
    ]
    return events_data


if __name__ == '__main__':
    args_in = sys.argv
    filenum = args_in[1]

    events_inputs = {
        'n_no_cue': 72,
        'n_double_cue': 72,
        'n_valid_cue': 54,
        'n_invalid_cue': 18,
        'cue_dur': 0.1,
        'cti_durs': [0.4],
        'stim_dur': 1,
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
    total_time = 13 * 55
    events_inputs['model_option'] = 'one_component'

    trials_psych_assess_map = {
        'cue_cti_stim_no_cue_400': '1',
        'cue_cti_stim_double_cue_400': '2',
        'cue_cti_stim_valid_cue_400': '3',
        'cue_cti_stim_invalid_cue_400': '4',
    }

    ntrials = (
        events_inputs['n_no_cue']
        + events_inputs['n_double_cue']
        + events_inputs['n_valid_cue']
        + events_inputs['n_invalid_cue']
    )
    unpermuted_trials = np.concatenate(
        [
            np.repeat('1', int(events_inputs['n_no_cue'] / 3)),
            np.repeat('2', int(events_inputs['n_double_cue'] / 3)),
            np.repeat('3', int(events_inputs['n_valid_cue'] / 3)),
            np.repeat('4', int(events_inputs['n_invalid_cue'] / 3)),
        ],
        axis=0,
    )
    avg_trial_repeats_info = calc_expected_run_num_by_chance(
        unpermuted_trials, nsims=5000
    )

    for val in ['1', '2', '3', '4']:
        avg_trial_repeats_info[f'{val}_run_counts'] = (
            avg_trial_repeats_info[f'{val}_run_counts'] * 3
        )
    avg_prob_given_last1, avg_prob_given_last2 = (
        calc_avg_prob_next_given_last1_and_last2(unpermuted_trials)
    )

    contrasts_one_component = {
        'double_400-no_cue_400': 'cue_cti_stim_double_cue_400-cue_cti_stim_no_cue_400',
        'valid_400-double_400': 'cue_cti_stim_valid_cue_400-cue_cti_stim_double_cue_400',
        'invalid_400-double_400': 'cue_cti_stim_invalid_cue_400-cue_cti_stim_double_cue_400',
        'all_task': '1/4*(cue_cti_stim_no_cue_400 + cue_cti_stim_double_cue_400 + cue_cti_stim_valid_cue_400 + cue_cti_stim_invalid_cue_400)',
    }

    nsim = 20000
    output_setting_one_comp, events_setting_one_comp = run_eff_sim(
        nsim,
        events_inputs,
        make_spatial_cueing_timings,
        contrasts_one_component,
        avg_trial_repeats_info,
        tr,
        total_time,
        trials_psych_assess_map,
        avg_prob_given_last1,
        avg_prob_given_last2,
        deriv=False,
    )

    output_setting_one_comp.to_csv(
        f'/home/users/jmumford/efficiency/output/spatial_cueing_output_{filenum}.csv'
    )
    with open(
        f'/home/users/jmumford/efficiency/output/spatial_cueing_events_{filenum}.pkl',
        'wb',
    ) as f:
        pickle.dump(events_setting_one_comp, f)
