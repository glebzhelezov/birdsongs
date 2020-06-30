#import boat
import boat_submission as boat 
import boat_draw_bursts
import important_settings

import glob
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
from scipy.special import binom
import numpy as np
from joblib import Parallel, delayed
from ete3 import Tree
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from glob import glob
from pickle import dump
from pickle import load
import io
import csv
import time
import sys

datafiles = glob("*.csv")
nwkfiles = glob("*.nwk")

all_analyses = []

n_pulses = 1

# Accept a different number of max pulses
if len(sys.argv) > 1:
    n_pulses = int(sys.argv[1])

if n_pulses == -1:
    important_settings.high_performance_flight_simulator()
    exit()

filename_prefix = "all_analyses_{}pulses_{}".format(n_pulses, time.time())
pickle_filename = filename_prefix + ".p"
output_filename = filename_prefix + "_output.dat"

for nwkfile in nwkfiles:
    all_analyses.append([])
    for datafile in datafiles:
        print(datafile)
        all_analyses[-1].append([])
        cur_analysis = all_analyses[-1][-1].append(nwkfile)
        cur_analysis = all_analyses[-1][-1].append(datafile)
        
        tree, data, error_vars = boat.load_treefile_datafile(nwkfile, datafile, return_errors=True)
        n_leaves, leaves = boat.number_leaves(tree)
        n_nodes, nodes, distances = boat.number_edges(tree)
    
        # find MLEs for BM
        bm_analysis = boat.bm_mle(data, tree, leaves, distances, error_variances=error_vars)
        # now find errors
        #_f = boat.bm_negloglkl_fun_withmu(data, tree, leaves, distances, error_vars)
        #hess = boat.numerical_hessian(lambda x:f(x[0], x[1], x[2]), bm_analysis[0], eps=0.000001)
        #bm_vars = np.linalg.inv(hess).diagonal()
        cur_analysis = all_analyses[-1][-1].append(bm_analysis)
        
        # find MLEs for pulsewe bought two
        pulse_analyses = boat.pulse_iterate_configs(data, tree, leaves, error_variances=error_vars, max_n=n_pulses)
        cur_analysis = all_analyses[-1][-1].append(pulse_analyses)
        
        # add # of pulses probability correction
        #for config in pulse_analyses:
        ##    n_pulses = sum([x[1] for x in config[2]])
        #    correction = 2*neg_log_pulse_prob(config[2], distances)
        ##    correction = 2*(-n_pulses * np.log(n_pulses/n_nodes) - (n_nodes - n_pulses)*np.log(1-n_pulses/n_nodes))
        #    config.append(config[-1] + correction)

        # sort by corrected aicc
        #pulse_analyses.sort(key=lambda x:x[-1], reverse=True)

        #pulse_aicc_config = pulse_analyses[0][3]

        #pulse_aicc = pulse_analyses[0][-1]
        #bm_aicc = bm_analysis[2]

with open(pickle_filename, "wb") as file:
    dump(all_analyses, file)
    
filename_translation = {
    'ind.co.var.bandwidth.pop.csv':'CV frequency bandwidth',
    'duration_by_pop.csv':'Duration',
    'ind.co.var.frequency.change.pop.csv':'CV frequency change',
    'ind.max.peak.frequency.pop.csv':'Max peak frequency',
    'log_duration_by_pop.csv':'Log duration',
    'ind.min.peak.frequency.pop.csv':'Min peak frequency',
    'log.ind.median.frequency.change.pop.csv':'Log median frequency change',
    'ind.median.gap.after.pop.csv':'Median pause duration',
    'ind.median.element.duration.pop.csv':'Median element duration',
    'ind.range.peak.frequency.pop.csv':'Range peak frequency',
    'ind.co.var.gap.after.pop.csv':'CV pause duration',
    'ind.co.var.peak.frequency.pop.csv':'CV peak frequency',
    'log.ind.elements.pop.csv':'Log number of elements',
    'log.ind.median.bandwidth.pop.csv':'Log median bandwidth',
    'ind.median.peak.frequency.pop.csv':'Median peak frequency',
}

with open(output_filename, "w") as output_file:
    csvwriter = csv.writer(output_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = [
        'nwk_file',
        'data_file',
        'BM_var_p',
        'BM_var_bm',
        'BM_mu',
        'BM_lkl',
        'BM_AICc',
        'PULSE_var_p',
        'PULSE_var_d',
        'PULSE_mu',
        'PULSE_config',
        'PULSE_lkl',
        'PULSE_AICc',
        'PULSE_AICc - BM_AICc'
    ]
    csvwriter.writerow(columns)

    for i in range(len(all_analyses)):
        for j in range(len(all_analyses[i])):
            bm_analysis = all_analyses[i][j][2]
            pulse_analyses = all_analyses[i][j][3]

            pulse_analyses.sort(key=lambda x:x[3]) # sort pulses by AICc scores
            best_pulse = pulse_analyses[0]

            nwkfile = all_analyses[i][j][0] # nwkfile
            datafile = all_analyses[i][j][1] # datafile

            bm_var_p = bm_analysis[0][0]
            bm_var_bm = bm_analysis[0][1]
            bm_mu = bm_analysis[0][2]
            bm_lkl = bm_analysis[1]
            bm_aicc = bm_analysis[2]

            pulse_var_p = best_pulse[0][0]
            pulse_var_d = best_pulse[0][1]
            pulse_mu = best_pulse[0][2]
            pulse_config = best_pulse[2]
            pulse_lkl= best_pulse[1]
            pulse_aicc = best_pulse[3]

            aicc_diff = pulse_aicc - bm_aicc

            row_to_write = [
                nwkfile,
                datafile,
                bm_var_p,
                bm_var_bm,
                bm_mu,
                bm_lkl,
                bm_aicc,
                pulse_var_p,
                pulse_var_d,
                pulse_mu,
                pulse_config,
                pulse_lkl,
                pulse_aicc,
                aicc_diff
            ]
            
            csvwriter.writerow(row_to_write)
            
            # draw a tree!
            if (aicc_diff < -2):
                print("Drawing tree for {} (Delta AICc < -2).".format(filename_translation[datafile]))
                fake_lkls = [] # for old drawing package
                
                for pulse_analysis in pulse_analyses:
                    pulse_var_p = pulse_analysis[0][0]
                    pulse_var_d = pulse_analysis[0][1]
                    pulse_mu = pulse_analysis[0][2]
                    pulse_config = pulse_analysis[2]
                    pulse_lkl= pulse_analysis[1]
                    pulse_aicc = pulse_analysis[3]
                    fake_lkls.append([pulse_lkl, 0, pulse_config, 0, pulse_aicc])
                
                boat_draw_bursts.plot_burst_tree(nwkfile, datafile, fake_lkls, extra_str="")
print("Done drawing trees!")

with open(pickle_filename, "rb") as f:
    all_analyses = load(f)

for analysis_index_i in range(len(all_analyses)):
    for analysis_index_j in range(len(all_analyses[analysis_index_i])):
        #print(analysis_index_i,analysis_index_j)
        bm_analysis = all_analyses[analysis_index_i][analysis_index_j][2]
        pulse_analyses = all_analyses[analysis_index_i][analysis_index_j][3]

        pulse_analyses.sort(key=lambda x:x[3]) # sort pulses by AICc scores
        best_pulse = pulse_analyses[0]

        nwkfile = all_analyses[analysis_index_i][analysis_index_j][0] # nwkfile
        datafile = all_analyses[analysis_index_i][analysis_index_j][1] # datafile
        
        bm_var_p = bm_analysis[0][0]
        bm_var_bm = bm_analysis[0][1]
        bm_mu = bm_analysis[0][2]
        bm_lkl = bm_analysis[1]
        bm_aicc = bm_analysis[2]

        pulse_var_p = best_pulse[0][0]
        pulse_var_d = best_pulse[0][1]
        pulse_mu = best_pulse[0][2]
        pulse_config = best_pulse[2]
        pulse_lkl= best_pulse[1]
        pulse_aicc = best_pulse[3]

        aicc_diff = pulse_aicc - bm_aicc
        pulse_var_d = best_pulse[0][1]
        pulse_mu = best_pulse[0][2]
        pulse_config = best_pulse[2]
        pulse_lkl= best_pulse[1]
        pulse_aicc = best_pulse[3]

        aicc_diff = pulse_aicc - bm_aicc

        
        if (aicc_diff < -2):
            print("Drawing graphs for {} (Delta AICc < -2).".format(filename_translation[datafile]))
            # load the data
            tree, data, errors = boat.load_treefile_datafile(nwkfile, datafile, return_errors=True)
            n_leaves, leaves = boat.number_leaves(tree)
            n_nodes, nodes, distances = boat.number_edges(tree)
            data = np.array(data)
            scaling = max(data) - min(data)

            data = (data - data.mean())/scaling
            errors = np.array(errors) / (scaling*scaling)
            
            # just to see effect of errors
            #errors = np.zeros(len(data))



            _f = boat.configuration_likelihood_neg_log_lkl_fun(
                boat.Models.PULSE_MU,
                data,
                tree,
                leaves,
                pulse_config,
                error_variances=errors
            )
            
            # Draw the likelihood surface
            fig = plt.figure()
            if len(pulse_config) > 0:
                _vf = np.vectorize(lambda x,y:np.exp(-_f([x,y])))
                x = np.linspace(1e-10, 0.04, 80)
                y = np.linspace(1e-10, 1, 80)
                X,Y = np.meshgrid(x, y)
                Z = _vf(X, Y)
                plt.pcolor(X, Y, Z)
                plt.colorbar()
                #print("config = {}, lkl = {}, aicc = {}".format(pulse_config, pulse_lkl, pulse_aicc))
                plt.scatter([pulse_var_p/(scaling*scaling)], [pulse_var_d/(scaling*scaling)], 50, label="fit")
                plt.title("Min AICc likelihood function\n{} {}".format(filename_translation[datafile], pulse_config))
                plt.xlabel("var_p")
                plt.ylabel("var_d")
                plt.legend()
                #plt.show()
                plt.savefig("likelihood_{}.png".format(datafile))
            else:
                #print("config = {}, lkl = {}, aicc = {}".format(pulse_config, pulse_lkl, pulse_aicc))
                _vf = np.vectorize(lambda x:np.exp(-_f([x, 0])))
                x = np.linspace(0, 0.1, 80)
                y = _vf(x)
                plt.title(datafile)
                plt.xlabel("var_p")
                plt.ylabel("likelihood")
                plt.plot(x,y)
                plt.scatter([pulse_var_p/(scaling*scaling)], [_vf(pulse_var_p/(scaling*scaling))], 50, label="fit")
                plt.title("Min AICc likelihood function\n{} {}".format(filename_translation[datafile], pulse_config))
                plt.legend()
                #plt.show()
                plt.savefig("likelihood_{}.png".format(datafile))
            
            plt.close()
            
            # Draw the mu(Sigma) surface
            fig = plt.figure()
            if len(pulse_config) > 0:
                _vf = np.vectorize(lambda x,y:_f([x,y], True)[1])
                x = np.linspace(1e-10, 0.04, 80)
                y = np.linspace(1e-10, 1, 80)
                X,Y = np.meshgrid(x, y)
                Z = _vf(X, Y)
                plt.pcolor(X, Y, Z)
                plt.colorbar()
                #print("config = {}, lkl = {}, aicc = {}".format(pulse_config, pulse_lkl, pulse_aicc))
                plt.scatter([pulse_var_p/(scaling*scaling)], [pulse_var_d/(scaling*scaling)], 50, label="fit")
                plt.title("MLE of $\mu$\n{} {}".format(filename_translation[datafile], pulse_config))
                plt.xlabel("var_p")
                plt.ylabel("var_d")
                plt.legend()
                #plt.show()
                plt.savefig("mean_{}.png".format(datafile))
            else:
                print("config = {}, lkl = {}, aicc = {}".format(pulse_config, pulse_lkl, pulse_aicc))
                _vf = np.vectorize(lambda x:_f([x, 0], True)[1])
                x = np.linspace(0, 0.1, 80)
                y = _vf(x)
                plt.title(datafile)
                plt.xlabel("var_p")
                plt.ylabel("likelihood")
                plt.plot(x,y)
                plt.scatter([pulse_var_p/(scaling*scaling)], [_vf(pulse_var_p/(scaling*scaling))], 50, label="fit")
                plt.title("MLE of $\mu$\n{} {}".format(filename_translation[datafile], pulse_config))
                plt.legend()
                #plt.show()
                plt.savefig("mean_{}.png".format(datafile))
            
            plt.close()

            ### Draw barcode
            grouped_leaf_ids = boat.group_ids(boat.group_by_regime(pulse_config, tree))

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)


            cm = plt.get_cmap('tab10')
            n_regimes = len(grouped_leaf_ids)
            for i in range(n_regimes):
                data_subset = [data[k] for k in grouped_leaf_ids[i]]
                error_subset = [errors[k] for k in grouped_leaf_ids[i]]
                plt.eventplot(data_subset, color=cm(i))
                plt.errorbar(data_subset, np.linspace(0.5, 1.5, len(error_subset)), yerr=None, xerr=np.sqrt(error_subset), fmt='none', color=cm(i))

            plt.xlim([-1,1])

            # Move left y-axis and bottim x-axis to centre, passing through (0,0)
            ax.spines['bottom'].set_position('center')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_visible(False)
            ax.set_title("Min AICc grouping\n{} {} ({} regimes)".format(filename_translation[datafile], pulse_config, len(grouped_leaf_ids)))
            #plt.show()
            plt.savefig("barcode_{}.png".format(datafile))
            
            plt.close()
