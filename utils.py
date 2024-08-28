import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import multivariate_normal, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix
from nilearn.glm.contrasts import expression_to_contrast_vector


def get_run_info(trial_order):
    '''
    Gives run counts for each condition.  
    E.g. aaabbabbb would find 2 runs of a of lengths 3 and 1 
        and 2 runs of b of lengths 2 and 3
    input: 
       trial_order.  Must be a 1D numpy array
    output:
       run_data_pd: data frame where each column corresponds to a trial-type
         and each row corresponds to a run length (index + 1 = run_length).  Columns
         are labeled accordingly
    '''
    if len(trial_order.shape) > 1:
        raise ValueError("Input trial order must be a 1D numpy array")
    n_trials = len(trial_order)
    trial_types, num_per_trial = np.unique(trial_order, return_counts=True)
    max_num_per_trial = np.max(num_per_trial)
    run_info_out = np.zeros((max_num_per_trial, len(trial_types)))
    trial_type_column_map = {trial: ind  for ind, trial in enumerate(trial_types)}
    counter = 0
    for i in range(1, n_trials):
        if trial_order[i] == trial_order[i-1]:
            counter = counter + 1
        if (trial_order[i] != trial_order[i-1]) or (i == (n_trials - 1)):
            run_info_out[counter, trial_type_column_map[trial_order[i-1]]] += 1
            counter = 0
    column_labels = [f'{trial_type}_run_counts' for trial_type in trial_types]
    run_data_pd = pd.DataFrame(run_info_out, columns=column_labels)
    run_data_pd.insert(0, 'run_length', np.array(run_data_pd.index + 1))
    return run_data_pd


def calc_expected_run_num_by_chance(unshuffled_cond_vector, nsims=5000):
    '''
    Randomly permutes trials many times and calculates average number of runs for each 
      possible run length for each condition
    Input:
       unshuffled_condition_vector: vector of conditions in any order, but with the correct number
                    of conditions and number of observations per condition represented
       nsim: How many simulations will be run
    Output:
      Average number of runs at each level of run length for each condition
    '''
    if len(unshuffled_cond_vector.shape) > 1:
        raise ValueError("Input trial order must be a 1D numpy array")
    run_info = []
    trial_order_copy = unshuffled_cond_vector.copy()
    for sim in range(nsims):
        np.random.shuffle(trial_order_copy)
        run_info.append(get_run_info(trial_order_copy))
    avg_run_info = sum(run_info)/nsims
    return avg_run_info



def FcCalc(trial_order, confoundorder=3):
    '''
    Compute confounding efficiency.
    This function is from NeuroDesign, but JM edited one formula to match the 
    Kao (2008) paper (commented below)

    :param confoundorder: To what order should confounding be protected
    :type confoundorder: integer
    '''
    unique_stim, counts = np.unique(trial_order, return_counts=True)
    n_trials = len(trial_order)
    n_stimuli = len(unique_stim)
    cond_label_dict = {label: count for count, label in enumerate(unique_stim)}
    P = counts/np.sum(counts) 
    Q = np.zeros([n_stimuli, n_stimuli, confoundorder])
    for n in range(len(trial_order)):
        for r in np.arange(1, confoundorder + 1):
            if n > (r - 1):
                Q[cond_label_dict[trial_order[n]], cond_label_dict[trial_order[n - r]], r - 1] += 1
    Qexp = np.zeros([n_stimuli, n_stimuli, confoundorder])
    for si in range(n_stimuli):
        for sj in range(n_stimuli):
            for r in np.arange(1, confoundorder + 1):
                Qexp[si, sj, r - 1] = P[si] * P[sj] * (n_trials  - r)  #JM edited this from Joke's code (ntrials +1? not ntrials-r)
    Qmatch = np.sum(abs(Q - Qexp))
    return Qmatch


def est_prob_runs_gte_2_trials(trial_order):
    '''
    Estimates the probability that a run (repeat of a specific trial) length
    is larger than or equal to 2.  E.g. aabbbabaaa has 5 runs, 3 of which are longer 
    than a single trial repeat, so the output would be 3/5.
    Input: 
    trial order: 1D numpy array of the trial order
    Output: 
    percent_runs_longer_than_1: Single value indicating the percent of the runs with 
        more than 1 repeated trial in a row
    '''
    run_info = get_run_info(trial_order)
    run_info_counts_only = run_info.loc[:, run_info.columns.str.endswith('_run_counts')]
    sum_counts_across_conditions = run_info_counts_only.sum(axis=1)
    percents = sum_counts_across_conditions/sum(sum_counts_across_conditions)
    percent_runs_longer_than_1 = sum(percents[run_info['run_length']> 1])
    return percent_runs_longer_than_1


def check_n_in_row(trial_order):
    '''
    Finds runs in trial order (blocks of repeats)
    and calculates how many runs occur for each run length
    input: trial order
    output a dictionary object containing: 
      run_length: integer indicating the length of a run
      num_runs: Number of runs with that length 
    '''
    n_trials = len(trial_order)
    counter = 0
    tracker = []
    for i in range(1, n_trials):
        if trial_order[i] == trial_order[i-1]:
            counter = counter + 1
        if (trial_order[i] != trial_order[i-1]) or i==(n_trials-1):
            tracker.append(counter)
            counter = 0
    run_length, num_runs = np.unique(tracker, return_counts=True)
    run_length = run_length + 1
    output = pd.DataFrame({'run_length': run_length, 'num_runs': num_runs})
    return output


def calc_run_num_diff_from_avg(trial_order, avg_run_info):
    '''
    Compares the average number of runs for each condition to the observed
      by calculating the sum of the absolute difference
    input:
      trial_order:  The trial order of interest
      avg_run_info: output from calc_expected_run_num_by_chance, the expected average
          number of runs for each length and each condition
    output:
      diff_stat:  The sum of the absolute difference between the observed and average counts
       smaller is better
    '''
    observed_trial_run_data = get_run_info(trial_order)
    avg_run = avg_run_info.drop(columns=['run_length'], inplace=False)
    observed_run = observed_trial_run_data.drop(columns=['run_length'], inplace=False)
    abs_diff_mat = np.abs(avg_run - observed_run)
    diff_stat = abs_diff_mat.sum().sum()
    return diff_stat


def calc_avg_prob_next_given_last1_and_last2(trial_order):
    cond_prob1_sum = []
    cond_prob2_sum = []
    nsim = 10000
    trial_order_copy = trial_order.copy()
    for i in range(nsim):
      np.random.shuffle(trial_order_copy)
      cond_prob1_loop, _, _ = calc_prob_next_given_last1(trial_order_copy)
      cond_prob2_loop, _, _ = calc_prob_next_given_last2(trial_order_copy)
      if i==0:
          cond_prob1_sum = cond_prob1_loop
          cond_prob2_sum = cond_prob2_loop
      else:
          cond_prob1_sum.iloc[:, cond_prob1_sum.columns != 'cond_val'] = \
            cond_prob1_sum.iloc[:, cond_prob1_sum.columns != 'cond_val'] + cond_prob1_loop.iloc[:, cond_prob1_sum.columns != 'cond_val']    
          cond_prob2_sum.iloc[:, cond_prob2_sum.columns != 'cond_pairs'] = \
            cond_prob2_sum.iloc[:, cond_prob2_sum.columns != 'cond_pairs'] + cond_prob2_loop.iloc[:, cond_prob2_sum.columns != 'cond_pairs']
    cond_prob_given_last1_avg = cond_prob1_sum.copy()
    cond_prob_given_last1_avg.iloc[:, cond_prob1_sum.columns != 'cond_val'] = \
        cond_prob_given_last1_avg.iloc[:, cond_prob1_sum.columns != 'cond_val']/nsim
    cond_prob_given_last2_avg = cond_prob2_sum.copy()
    cond_prob_given_last2_avg.iloc[:, cond_prob2_sum.columns != 'cond_pairs'] = \
        cond_prob_given_last2_avg.iloc[:, cond_prob2_sum.columns != 'cond_pairs']/nsim
    return cond_prob_given_last1_avg, cond_prob_given_last2_avg


def calc_prob_next_given_last1(trial_order, cond_prob_given_last1_avg=None):
    '''
    Calculates the conditional probability of the next trial being a specific type 
    given the current trial type
    Input:
      trial_order:  The trial order being studied
    Output:
      output_prob_df: Data frame that includes the conditional probabilities
      max_prob:  The maximum conditional probability found (max of output_prob_df)
    '''
    trial_order_str = ''.join(trial_order)
    cond_list = np.unique(trial_order)
    output_prob = {f'pr({cond}|cond_val)':[] for cond in cond_list}    
    for single in cond_list:
        single_locations = [m.start() for m in re.finditer(f'(?={single})', trial_order_str)]
        after_single_location = np.array(single_locations) + 1
        after_single_location = after_single_location[after_single_location < len(trial_order)]
        num_single = len(after_single_location)
        conds_after_singles = [trial_order_str[val] for val in after_single_location]
        cond_after, counts = np.unique(conds_after_singles, return_counts=True)
        for cond in cond_list:
            if cond in cond_after:
                output_prob[f'pr({cond}|cond_val)'].append(counts[cond_after == cond][0]/num_single)
            if cond not in cond_after:
                output_prob[f'pr({cond}|cond_val)'].append(0)
    output_prob_df = pd.DataFrame(output_prob)
    max_prob = output_prob_df.values.max()
    output_prob_df.insert(0, 'cond_val', cond_list)
    output_prob_df = output_prob_df.fillna(0)
    if cond_prob_given_last1_avg is not None:
        sum_abs_diff_w_avg_prob = np.sum(np.abs(cond_prob_given_last1_avg.drop(['cond_val'], axis=1) - 
                                         output_prob_df.drop(['cond_val'], axis=1)).values)
    else:
        sum_abs_diff_w_avg_prob = None
    return output_prob_df, max_prob, sum_abs_diff_w_avg_prob


def calc_prob_next_given_last2(trial_order, cond_prob_given_last2_avg=None):
    '''
    Calculates the conditional probability of the next trial being a specific type 
    given the current trial type.  Note, the result is set to nan if a condition pair
    only occurs 1 time (since the max probability of 1 given last 2 will be 1 in this case
    and that can easily occur when there are more trial types)
    Input:
      trial_order:  The trial order being studied
    Output:
      output_prob_df: Data frame that includes the conditional probabilities
      max_prob:  The maximum conditional probability found (max of output_prob_df)
    '''
    trial_order_str = ''.join(trial_order)
    cond_list = np.unique(trial_order)
    ncond = len(cond_list)
    all_pairs = []
    for i in range(ncond):
        for j in range(ncond):
          all_pairs.append(f'{cond_list[i]}{cond_list[j]}')
    output_prob = {f'pr({cond}|cond_pairs)':[] for cond in cond_list}    
    for pair in all_pairs:
        pair_locations = np.array([m.start() for m in re.finditer(f'(?={pair})', trial_order_str)])
        pair_locations = pair_locations[pair_locations < (len(trial_order) - 2)]
        if len(pair_locations)>1:
            after_pair_location = np.array(pair_locations) + 2
            after_pair_location = after_pair_location[after_pair_location < len(trial_order)]
            num_pairs = len(after_pair_location)
            conds_after_pairs = [trial_order_str[val] for val in after_pair_location]
            cond_after, counts = np.unique(conds_after_pairs, return_counts=True)
            for cond in cond_list:
                if cond in cond_after:
                    output_prob[f'pr({cond}|cond_pairs)'].append(counts[cond_after == cond][0]/num_pairs)
                if cond not in cond_after:
                    output_prob[f'pr({cond}|cond_pairs)'].append(0)
        else:
            for cond in cond_list:
                output_prob[f'pr({cond}|cond_pairs)'].append(np.nan)
    output_prob_df = pd.DataFrame(output_prob)
    max_prob = np.nanmax(output_prob_df)
    output_prob_df.insert(0, 'cond_pairs', all_pairs)
    output_prob_df = output_prob_df.fillna(0)
    if cond_prob_given_last2_avg is not None:
        sum_abs_diff_w_avg_prob = np.sum(np.abs(cond_prob_given_last2_avg.drop(['cond_pairs'], axis=1) - 
                                         output_prob_df.drop(['cond_pairs'], axis=1)).values)
    else:
        sum_abs_diff_w_avg_prob = None
    return output_prob_df, max_prob, sum_abs_diff_w_avg_prob


def pred_second_half_from_first(trial_order):
    '''
    Calculate predicted values for the second half of the data based on 
     Pr(next | last 2) estimates from the first half of the data.  When multiple
     conditions share the max probability, one condition is randomly chosen as the prediction
     Otherwise the condition with max(pr(next | last2)) is chosen as the predicted value.
    Input:
      trial_order:  The trial order of interest
    Output:
      Percent of 2nd half of data that was correctly predicted
    '''
    half_trials = int(len(trial_order)/2)
    probs_last2, _ = calc_prob_next_given_last2(trial_order[:half_trials])
    probs_last2_nopairs = probs_last2.drop(columns=['cond_pairs'])
    prediction_values = {}
    for row in range(probs_last2_nopairs.shape[0]):
        pair_name = probs_last2['cond_pairs'][row]
        row_max = np.max(probs_last2_nopairs.loc[row,:])
        max_location = probs_last2_nopairs.loc[row]==row_max
        colnames_max = probs_last2_nopairs.columns[max_location]
        if len(colnames_max)>0:
            conds_max = [re.search(r'pr\(([a-z]*)|cond_pairs\)', val).group(1) for val in colnames_max]
            prediction_values[pair_name] = conds_max
    match = []
    pred_list = []
    actual_val = []
    for i in range(half_trials+1, len(trial_order)-1):
        cur2 = trial_order[(i-1):(i+1)]
        next_after_cur2 = trial_order[i+1]
        actual_val.append(next_after_cur2)
        cur2_str = ''.join(list(cur2))
        if cur2_str not in prediction_values.keys():
            match.append(False)
            pred_list.append(False)
        if cur2_str in prediction_values.keys():
            pred_val = np.random.choice(prediction_values[cur2_str], size=1)[0]
            pred_list.append(pred_val)
            match.append(pred_val == next_after_cur2)
    return np.mean(match)


def sample_shifted_truncated_exponential(exp_lam, T, shift, nsamp):
    '''
    Samples nsamp values from a shifted and truncated exponential distribution
    Input:
      exp_lam: rate parameter for exponential (untruncated mean is 1/lambda)
      T: Truncation value (before shift)
      shift: shift value
      nsamp:  The number of samples desired
    Output:
      samples: samples from the distribution
      theoretical_mean:  The theoretical mean of the distribution (Likely smaller than 1/lambda,
        depending on how low truncation value is)
    '''
    theoretical_mean = shift + 1/exp_lam - T/(np.exp(exp_lam*T)-1)
    R = np.random.uniform(0, 1, nsamp) * (1-np.exp(-T*exp_lam))
    samples = -1*np.log(1-R) * 1/exp_lam + shift
    return samples, theoretical_mean


def make_contrast_matrix(contrasts, desmat):
    '''
    Creates a contrast matrix from a dictionary of contrasts written in expression form
    using nilearn's expression_to_contrast_vector
    input:
        contrasts:  Dictionary of contrasts written in expression form.  The expressions
           consist of the contrast formulae using the column names from desmat
        desmat:  The design matrix for the fMRI model
    output:
        contrast_matrix:  A n_contrast x n_params numpy array of contrasts
    '''
    contrast_matrix = []
    for key, val in contrasts.items():
        contrast_matrix.append(expression_to_contrast_vector(val, desmat.columns))
    contrast_matrix = np.array(contrast_matrix)
    return contrast_matrix


def est_eff_and_vif(events, tr, max_total_time, contrasts, time_past_last_offset=20, deriv=True):
    '''
    Builds design matrix and estimates efficiency for all contrasts
    '''
    if deriv:
        hrf_model_val = 'spm + derivative'
    else:
        hrf_model_val = 'spm'
    scan_cutoff = np.floor((events.onset.values[-1:][0] + events.duration.values[-1:][0] + time_past_last_offset))
    frame_times = np.arange(0, scan_cutoff, tr)
    desmat = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=hrf_model_val,
        drift_model='cosine')
    contrast_matrix = make_contrast_matrix(contrasts, desmat)
    cov_mat = contrast_matrix @ np.linalg.inv(desmat.transpose()@desmat) @ contrast_matrix.transpose()
    var_vec = np.diag(cov_mat)
    eff_vec = 1/var_vec
    contrast_names = list(contrasts.keys())
    eff_out = {contrast_names[i]: eff_vec[i] for i in range(len(eff_vec)) }

    vifs = get_all_contrast_vif(desmat, contrasts)
    return eff_out, vifs, desmat
    

def est_psych_fitness(trial_type, avg_run_info, cond_prob_given_last1_avg, cond_prob_given_last2_avg):
    '''
    Estimate all of the psychological fitness measures for a given trial order
    '''
    output = {}
    output['kao_measure'] = FcCalc(trial_type, confoundorder=3)
    output['prob_runs_gte_2']  = est_prob_runs_gte_2_trials(trial_type)
    output['run_num_diff_from_avg'] = calc_run_num_diff_from_avg(trial_type, avg_run_info)
    _, output['prob_next_given_last1'], output['sum_abs_diff_prob_next_given_last1'] = calc_prob_next_given_last1(trial_type, cond_prob_given_last1_avg)
    _, output['prob_next_given_last2'], output['sum_abs_diff_prob_next_given_last2'] = calc_prob_next_given_last2(trial_type, cond_prob_given_last2_avg)
    #output['pred_second_half_from_first'] = pred_second_half_from_first(trial_type)
    return output



def est_vif(desmat):
    '''
    General variance inflation factor estimation.  Calculates VIF for all 
    regressors in the design matrix
    input:
        desmat: design matrix.  Intercept not required.
    output:
      vif_data: Variance inflation factor for each regressor in the design matrix
                generally goal is VIF<5
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    desmat_with_intercept = desmat.copy()
    desmat_with_intercept['intercept'] = 1
    vif_data = {}
    vif_data['regressor'] = []
    vif_data['VIF'] = []
    for i, name in enumerate(desmat_with_intercept.columns):
        if desmat_with_intercept[name].std() != 0:
            vif_data['regressor'].append(name)
            vif_data['VIF'].append(variance_inflation_factor(desmat_with_intercept.values, i))
    vif_data = pd.DataFrame(vif_data)
    return vif_data


def get_eff_reg_vif(desmat, contrast):
    '''
    The goal of this function is to estimate a variance inflation factor for a contrast.
    This is done by extending the effective regressor definition from Smith et al (2007)
    Meaningful design and contrast estimability (NeuroImage).  Regressors involved
    in the contrast estimate are rotated to span the same space as the original space
    consisting of the effective regressor and and an orthogonal basis.  The rest of the 
    regressors are unchanged.
    input:
        desmat: design matrix.  Assumed to be a pandas dataframe with column  
             headings which are used define the contrast of interest
        contrast: a single contrast defined in string format
    output:
        vif: a single VIF for the contrast of interest  
    '''
    from scipy.linalg import null_space
    from nilearn.glm.contrasts import expression_to_contrast_vector
    contrast_def = expression_to_contrast_vector(contrast, desmat.columns)
    des_nuisance_regs = desmat[desmat.columns[contrast_def == 0]]
    des_nuisance_regs = des_nuisance_regs.reset_index(drop=True)
    des_contrast_regs = desmat[desmat.columns[contrast_def != 0]]

    con = np.atleast_2d(contrast_def[contrast_def != 0])
    con2_t = null_space(con)
    con_t = np.transpose(con)
    x = des_contrast_regs.copy().values
    q = np.linalg.pinv(np.transpose(x)@ x)
    f1 = np.linalg.pinv(con @ q @ con_t)
    pc = con_t @ f1 @ con @ q
    con3_t = con2_t - pc @ con2_t
    f3 = np.linalg.pinv(np.transpose(con3_t) @ q @ con3_t)
    eff_reg = x @ q @ np.transpose(con) @ f1
    eff_reg = pd.DataFrame(eff_reg, columns = [contrast])

    other_reg = x @ q @ con3_t @ f3 
    other_reg_names = [f'orth_proj{val}' for val in range(other_reg.shape[1])]
    other_reg = pd.DataFrame(other_reg, columns = other_reg_names)

    des_for_vif = pd.concat([eff_reg, other_reg, des_nuisance_regs], axis = 1)
    vif_dat = est_vif(des_for_vif)
    vif_dat.rename(columns={'regressor': 'contrast'}, inplace=True)
    vif_output = vif_dat[vif_dat.contrast == contrast]
    return vif_output


def get_all_contrast_vif(desmat, contrasts):
    '''
    Calculates the VIF for multiple contrasts
    input:
        desmat: design matrix.  Pandas data frame, column names must 
                be used in the contrast definitions
        contrasts: A dictionary of contrasts defined in string format
    output:
        vif_contrasts: Data frame containing the VIFs for all contrasts
    '''
    vifs = {}
    for key, item in contrasts.items():
        vif_out = get_eff_reg_vif(desmat, item)
        vifs[key] = vif_out['VIF'][0]
    return vifs   


def run_eff_sim(nsim, events_inputs, make_timings_function, contrasts, 
                avg_trial_repeats_info, tr, max_total_time, trials_psych_assess_map,
                cond_prob_given_last1_avg, cond_prob_given_last2_avg,
                time_past_last_offset=20, deriv=True, est_psych=True, name_swap=None):
    '''
    Runs nsim randomly created stop signal designs through efficiency/vif/psych fitness
    measures and outputs results
    Input: 
      nsim:  Number of simulated design matrices
    Output:
      output: Pandas data frame with each fitness measure for each design
      events_all:  The events data for each design matrix
    '''
    psych_variables = ['kao_measure', 'prob_runs_gte_2', 'run_num_diff_from_avg',
                       'prob_next_given_last1', 'prob_next_given_last2',
                       'sum_abs_diff_prob_next_given_last1', 'sum_abs_diff_prob_next_given_last2']
    output = {f'eff_{contrast_name}':[] for contrast_name in contrasts.keys()}
    for contrast_name in contrasts.keys():
        output[f'vif_{contrast_name}'] = [] 
    output['scan_length'] = []
    if est_psych==True:
        for var in psych_variables:
            output[var] = []

    all_events = []

    for sim in range(nsim):
        events = make_timings_function(**events_inputs)
        scan_cutoff = np.floor((events.onset.values[-1:][0] + events.duration.values[-1:][0] + time_past_last_offset))
        if scan_cutoff > max_total_time:
            print('WARNING:  You need to increase the total time to fit all trials \n'
                'estimates from this simulation set should be discarded')
        eff_vals, vifs, _ = est_eff_and_vif(events, tr, max_total_time, contrasts,  time_past_last_offset=time_past_last_offset,
                                            deriv=deriv)
        if est_psych == True:
            trials_psych_assess = events['trial_type'][events['trial_type'].isin(trials_psych_assess_map)]
            trials_psych_assess_coded = np.array(trials_psych_assess.replace(trials_psych_assess_map))
            psych_assess = est_psych_fitness(trials_psych_assess_coded, avg_trial_repeats_info, 
                                             cond_prob_given_last1_avg, cond_prob_given_last2_avg)
            for key, val in psych_assess.items():
                output[key].append(val)
        for key, val in eff_vals.items():
            output[f'eff_{key}'].append(val)
        for key, val in vifs.items():
            output[f'vif_{key}'].append(val)
        if name_swap:
            events['trial_type'] = events['trial_type'].replace(name_swap, regex=True)
        all_events.append(events)
        output['scan_length'].append(
            events.onset.values[-1:][0] + events.duration.values[-1:][0] + time_past_last_offset)
    return pd.DataFrame(output), all_events