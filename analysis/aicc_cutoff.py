# Find AICc cutoffs for data generated using fit values

import glob
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
from scipy.special import binom
import numpy as np
import boat_legacy_matrix as boat_legacy
import boat_submission
from joblib import Parallel, delayed
from ete3 import Tree
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from glob import glob
import pickle
import time
import sys

print(
    """Generating synthetic data + fitting models to it. Usage:

$ {} pickle_filename.p
----------------------------------------------------
""".format(
        sys.argv[0]
    )
)


tree_filename = "UPGMA_song_match_2_Shen_to_South.nwk"
pickle_filename = sys.argv[1]  # "all_analyses_4pulses.p"
tree = Tree(tree_filename)


def aicc_diff_from_bm_data(
    tree,
    n_datasets,
    mu_min=-1,
    mu_max=1,
    var_bm_min=0.7,
    var_bm_max=4,
    var_p_min=0,
    var_p_max=0.1,
    relative_meas_error_min=0,
    relative_meas_error_max=0.3,
    n_samples_min=2,
    n_samples_max=30,
    n_search_pulses_max=2,
):
    """Generate sequence of AICc difference between BM and pulse model, using artificial BM data.
    
    arguments=
    tree,
    n_datasets,
    mu_min=-1,
    mu_max=1,
    var_bm_min=0.7,
    var_bm_max=4,
    var_p_min=0,
    var_p_max=0.1,
    relative_meas_error_min=0,
    relative_meas_error_max=0.3,
    n_samples_min=2,
    n_samples_max=30,
    n_search_pulses_max=2,"""
    tree_copy = tree.copy()
    n_leaves, leaves = boat_submission.number_leaves(tree_copy)
    n_nodes, nodes, distances = boat_submission.number_edges(tree_copy)

    bm_aicc_minus_pulse_aiccs = []

    for _ in range(n_datasets):
        # GENERATE RANDOM BM DATA
        rand_mean = np.random.uniform(mu_min, mu_max)
        rand_mean_vec = rand_mean * np.ones(n_leaves)
        rand_var_bm = np.random.uniform(var_bm_min, var_bm_max)
        rand_var_p = np.random.uniform(var_p_min, var_p_max)

        rand_cov_matrix = boat_legacy.build_bm_cov_matrix(
            tree, leaves, rand_var_p, rand_var_bm
        )

        real_mean = np.random.multivariate_normal(rand_mean_vec, rand_cov_matrix)

        # Now generata random measurement errors
        error_vars = np.zeros(len(real_mean))
        data = np.zeros(len(real_mean))
        for i in range(len(real_mean)):
            relative_meas_error = np.random.uniform(
                relative_meas_error_min, relative_meas_error_max
            )
            scale = np.abs((relative_meas_error * real_mean[i]))
            num_artificial_measurements = np.random.randint(
                n_samples_min, n_samples_max
            )
            samples = np.random.normal(
                loc=real_mean[i], scale=scale, size=num_artificial_measurements
            )
            error_vars[i] = np.var(samples)
            data[i] = np.mean(samples)

        data = np.array(data)

        # find MLEs for pulse and BM models
        bm_analysis = boat_submission.bm_mle(
            data, tree_copy, leaves, distances, error_variances=error_vars
        )
        pulse_analyses = boat_submission.pulse_iterate_configs(
            data,
            tree_copy,
            leaves,
            error_variances=error_vars,
            max_n=n_search_pulses_max,
        )

        bm_aicc_minus_pulse_aiccs.append(bm_analysis[2] - pulse_analyses[0][3])

    return bm_aicc_minus_pulse_aiccs


def aicc_diff_from_pulse_data(
    tree,
    n_datasets,
    mu_min=-1,
    mu_max=1,
    var_d_min=1,
    var_d_max=3,
    var_p_min=0,
    var_p_max=0.1,
    n_generate_pulses_max=2,
    relative_meas_error_min=0,
    relative_meas_error_max=0.3,
    n_samples_min=2,
    n_samples_max=30,
    n_search_pulses_max=2,
):
    """Generate sequence of AICc difference between BM and pulse model, using artificial pulse data.
    
    arguments =
    tree,
    n_datasets,
    mu_min=-1,
    mu_max=1,
    var_d_min=1,
    var_d_max=3,
    var_p_min=0,
    var_p_max=0.1,
    n_generate_pulses_max=2,
    relative_meas_error_min=0,
    relative_meas_error_max=0.3,
    n_samples_min=2,
    n_samples_max=30,
    n_search_pulses_max=2,"""
    tree_copy = tree.copy()
    n_leaves, leaves = boat_submission.number_leaves(tree_copy)
    n_nodes, nodes, distances = boat_submission.number_edges(tree_copy)

    bm_aicc_minus_pulse_aiccs = []

    for _ in range(n_datasets):
        # Store how many pulses on ith edge
        rand_pulses = np.zeros(n_nodes, dtype=int)
        # Generate a random number of pulses. Note we're sampling uniformly,
        # i.e. many of our samples will be just 1 pulse, even though in principle
        # there are many more 2- or 3-pulse possibilities.
        n_pulses = np.random.randint(1, n_generate_pulses_max + 1)
        rand_pulse_positions = np.random.choice(n_nodes, size=n_pulses)

        for pulse_index in rand_pulse_positions:
            rand_pulses[pulse_index] += 1

        # Random parameters
        rand_mean = np.random.uniform(mu_min, mu_max)
        rand_mean_vec = rand_mean * np.ones(n_leaves)
        rand_var_d = np.random.uniform(var_d_min, var_d_max)
        rand_var_p = np.random.uniform(var_p_min, var_p_max)

        rand_cov_matrix = boat_legacy.build_cov_matrix(
            tree, leaves, rand_pulses, rand_var_p, rand_var_d
        )

        real_mean = np.random.multivariate_normal(rand_mean_vec, rand_cov_matrix)

        # Now generata random measurement errors
        error_vars = np.zeros(len(real_mean))
        data = np.zeros(len(real_mean))
        for i in range(len(real_mean)):
            relative_meas_error = np.random.uniform(
                relative_meas_error_min, relative_meas_error_max
            )
            scale = np.abs((relative_meas_error * real_mean[i]))
            num_artificial_measurements = np.random.randint(
                n_samples_min, n_samples_max
            )
            samples = np.random.normal(
                loc=real_mean[i], scale=scale, size=num_artificial_measurements
            )
            error_vars[i] = np.var(samples)
            data[i] = np.mean(samples)

        data = np.array(data)

        # find MLEs for pulse and BM models
        bm_analysis = boat_submission.bm_mle(
            data, tree_copy, leaves, distances, error_variances=error_vars
        )
        pulse_analyses = boat_submission.pulse_iterate_configs(
            data,
            tree_copy,
            leaves,
            error_variances=error_vars,
            max_n=n_search_pulses_max,
        )

        bm_aicc_minus_pulse_aiccs.append(bm_analysis[2] - pulse_analyses[0][3])

    return bm_aicc_minus_pulse_aiccs


# Load analysis results, generate synthetic data.

# with open("all_analyses_4pulses.p", "rb") as f:
with open(pickle_filename, "rb") as f:
    all_analyses = pickle.load(f)

n_analyses = len(all_analyses[0])

data_filenames = []
bm_fits = []
pulse_fits = []
bm_differences = []
pulse_differences = []

# record all the values
for analysis in all_analyses[0]:
    data_file = analysis[1]
    bm_fit = analysis[2][0]
    pulse_fit = analysis[3][0][0]

    data_filenames.append(data_file)
    bm_fits.append(bm_fit)
    pulse_fits.append(pulse_fit)

tree_filename = "UPGMA_song_match_2_Shen_to_South.nwk"
tree = Tree(tree_filename)
# number of datasets to generate
n_datasets = 2 #1000
relative_meas_error_min = 0
relative_meas_error_max = 0.3
n_samples_min = 2
n_samples_max = 30
n_search_pulses_max = 1
n_generate_pulses_max = 3
alpha = 0.25

cutoff_fraction = 0.05

print("# Cutoff DeltaAICc for cutoff = {}".format(cutoff_fraction))
print("# First DeltaAICc is for BM-generated data, second is for pulse-generated data.")
print("filename,BM_generated,pulse_generated")

for i in range(n_analyses):
    ######
    ######
    ###### Compute AICc differences for BM-generated data
    ######
    ######
    fit_mu = bm_fits[i][0]
    fit_var_p = bm_fits[i][1]
    fit_var_bm = bm_fits[i][2]
    # generate data using +/- 25% of BM parameters
    mu_min = (1 - alpha) * fit_mu
    mu_max = (1 + alpha) * fit_mu
    var_bm_min = (1 - alpha) * fit_var_bm
    var_bm_max = (1 + alpha) * fit_var_bm
    var_p_min = (1 - alpha) * fit_var_p
    var_p_max = (1 + alpha) * fit_var_p

    differences = aicc_diff_from_bm_data(
        tree,
        n_datasets,
        mu_min,
        mu_max,
        var_bm_min,
        var_bm_max,
        var_p_min,
        var_p_max,
        relative_meas_error_min,
        relative_meas_error_max,
        n_samples_min,
        n_samples_max,
        n_search_pulses_max,
    )

    bm_differences.append(differences)

    # Find the DeltaAICc cutoff
    temp = differences.copy()
    temp.sort()
    cutoff_bm = temp[int(cutoff_fraction * len(temp))]

    ######
    ######
    ###### Compute AICc differences for pulse-generated data
    ######
    ######
    fit_mu = pulse_fits[i][0]
    fit_var_p = pulse_fits[i][1]
    fit_var_d = pulse_fits[i][2]
    # generate data using +/- 25% of pulse parameters
    mu_min = (1 - alpha) * fit_mu
    mu_max = (1 + alpha) * fit_mu
    var_d_min = (1 - alpha) * fit_var_d
    var_d_max = (1 + alpha) * fit_var_d
    var_p_min = (1 - alpha) * fit_var_p
    var_p_max = (1 + alpha) * fit_var_p

    differences = aicc_diff_from_pulse_data(
        tree,
        n_datasets,
        mu_min,
        mu_max,
        var_d_min,
        var_d_max,
        var_p_min,
        var_p_max,
        n_generate_pulses_max,
        relative_meas_error_min,
        relative_meas_error_max,
        n_samples_min,
        n_samples_max,
        n_search_pulses_max,
    )

    pulse_differences.append(differences)

    # Find the DeltaAICc cutoff
    temp = differences.copy()
    temp.sort()
    cutoff_pulse = temp[int((1 - cutoff_fraction) * len(temp))]

    print(data_filenames[i], cutoff_bm, cutoff_pulse, sep=",")

param_labels = [
    "n_synthetic_datasets",
    "rel_meas_error_min",
    "rel_meas_error_max",
    "n_samples_min",
    "n_samples_max",
    "n_search_pulses_max",
    "n_generate_pulses_max",
    "alpha",
]

param_vals = [
    n_datasets,
    relative_meas_error_min,
    relative_meas_error_max,
    n_samples_min,
    n_samples_max,
    n_search_pulses_max,
    n_generate_pulses_max,
    alpha,
]

mc_parameters = list(zip(param_labels, param_vals))

labels = [
    "monte_carlo_params",
    "data_filenames",
    "bm_generated_aicc_diff",
    "pulse_generated_aicc_diff",
]

all_differences = [
    labels,
    mc_parameters,
    data_filenames,
    bm_differences,
    pulse_differences,
]

filename = "generated_aicc_differences_{}.p".format(time.time_ns())

with open(filename, "wb") as f:
    pickle.dump(all_differences, f)
