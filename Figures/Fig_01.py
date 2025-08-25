#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 17:20:05 2025

@author: me-tcoons
"""

import numpy as np
import matplotlib.pyplot as plt
from mxmc import Optimizer
from mxmc import Estimator
import torch
import sys
import os
module_directory = os.path.abspath('../src/') 
sys.path.insert(0, module_directory) 
import model_utils_mono

def calculate_cov_delta_terms(estimator, true_cov):
        k_0 = estimator._allocation.get_k0_matrix()
        k = estimator._allocation.get_k_matrix()
        cov_q_delta = k_0 * true_cov[0, 1:]
        cov_delta_delta = k * true_cov[1:, 1:]
        return cov_delta_delta, cov_q_delta
    

# Parameters for initialization
x = torch.tensor([.5])  # Replace with your actual input data
n_models = 4           # Number of models
w = torch.tensor([[1.],[0.1],[0.01],[0.001]]) # model costs
w = w.detach().numpy().flatten()
pilot_cost = w.sum()
total_budget = 200*pilot_cost
start_fine = 4
end_fine = 20
start_coarse = end_fine+1
end_coarse=191
seed = 42
batches = np.concatenate((np.linspace(start_fine,end_fine,num=int((end_fine-start_fine+1)),dtype=int),
                          np.linspace(start_coarse,end_coarse,num=int((end_coarse-start_coarse+1)/5),dtype=int)))


# define best-case oracle estimator
oracle_cov = model_utils_mono.sig_func().detach().numpy()
oracle_budget = total_budget
variance_results = dict()
sample_allocation_results = dict()
mxmc_optimizer = Optimizer(w, oracle_cov)
algorithms = list(Optimizer.get_algorithm_names())
for algorithm in ['wrdiff']:#algorithms:
    if algorithm=='mfmc' or algorithm=='mlmc':
        variance_results[algorithm] = 999999.
    else:
        opt_result = mxmc_optimizer.optimize(algorithm, oracle_budget)
        variance_results[algorithm] = opt_result.variance
        sample_allocation_results[algorithm] = opt_result.allocation
        #print("{} method avg. variance: {}".format(algorithm, variance_results[algorithm]))
best_method_oracle = min(variance_results, key=variance_results.get)
sample_allocation = sample_allocation_results[best_method_oracle]
#print("best method: "+best_method_pilot)
estimator_oracle = Estimator(sample_allocation, oracle_cov)
sample_allocation_oracle = sample_allocation
var_oracle = variance_results[best_method_oracle]

iw_projections = True
method = 'wrdiff'
var_batches_projected = np.zeros((len(batches),))
var_batches_projected_iw = np.zeros((len(batches),))
var_batches_actual = np.zeros((len(batches),))
var_batches_actual_iw = np.zeros((len(batches),))
i=0
vars_n_star_iw=[]
vars_n_star=[]
seed=35
for batch in batches:
    y_pilot = model_utils_mono.model_eval_seed(x, batch, seed=seed).T
    c_hat = torch.cov(y_pilot, correction=1).detach().numpy()
    S = torch.cov(y_pilot, correction=0).detach().numpy()
    pilot_budget = batch*pilot_cost
    batch_budget = total_budget - pilot_budget
    
    # projected variance
    variance_results = dict()
    sample_allocation_results = dict()
    mxmc_optimizer = Optimizer(w, c_hat)
    algorithms = list(Optimizer.get_algorithm_names())
    for algorithm in [method]:#algorithms:
        if algorithm!=algorithm=='mfmc' or algorithm=='mlmc':
            variance_results[algorithm] = 999999.
        else:
            opt_result = mxmc_optimizer.optimize(algorithm, batch_budget)
            variance_results[algorithm] = opt_result.variance
            sample_allocation_results[algorithm] = opt_result.allocation
            #print("{} method avg. variance: {}".format(algorithm, var_mean_results[algorithm]))
    best_method_pilot = min(variance_results, key=variance_results.get)
    sample_allocation_pilot = sample_allocation_results[best_method_pilot]
    estimator_pilot = Estimator(sample_allocation_pilot, c_hat)
    var_batches_projected[i] = variance_results[best_method_pilot]
    
    # actual variance under oracle_cov
    cov_delta_delta, cov_q_delta = calculate_cov_delta_terms(estimator_pilot, oracle_cov)
    alpha = estimator_pilot._calculate_alpha()
    var_sf = oracle_cov[0,0]/sample_allocation_pilot.get_number_of_samples_per_model()[0]
    var_batches_actual[i] = var_sf + alpha@cov_delta_delta@alpha + 2*alpha@cov_q_delta
    i+=1 


#%% producing error vs batch plot
fig, ax = plt.subplots()
var_mc = oracle_cov[0,0]/(total_budget/w[0])
ax.plot(batches, var_oracle*np.ones((len(batches),)), linestyle='dashed', linewidth=2, c='gray', label='Oracle')
ax.plot(batches, var_mc*np.ones((len(batches),)), linestyle='dashed', linewidth=2, c='black', label='Single-Fidelity')
ax.plot(batches,var_batches_actual, marker='.', markersize=6, linewidth=2, c='red', label='Pilot Sampling ACV')
#ax.scatter(batches[np.argmin(var_batches_actual)], [var_batches_actual[np.argmin(var_batches_actual)]], marker='*', label=r'$n^{*}$', c='blue', s=80, zorder=99)
ax.set_xlabel('Pilot Sampling Size')
#ax.set_title('Effect of Pilot Sampling on Estimator Errors')
ax.set_ylabel('Estimator Variance')
ax.set_yscale('log')
ax.legend(loc='lower right')

#%% showing inaccuracy of projections
fig2, ax2 = plt.subplots()
n_fine = len(np.linspace(start_fine,end_fine,num=int((end_fine-start_fine+1)),dtype=int))
ax2.plot(batches[:n_fine], var_oracle*np.ones((len(batches[:n_fine]),)), linestyle='dashed', linewidth=2, c='gray', label='Oracle')
ax2.plot(batches[:n_fine], var_mc*np.ones((len(batches[:n_fine]),)), linestyle='dashed', linewidth=2, c='black', label='Single-Fidelity')
ax2.plot(batches[:n_fine],var_batches_projected[:n_fine], marker='^', markersize=5, linewidth=2, c='coral', label='Predicted')
ax2.plot(batches[:n_fine],var_batches_actual[:n_fine], marker='.', markersize=6, linewidth=2, c='red', label='Actual')
ax2.set_xlabel('Pilot Sampling Size')
# Set x-ticks manually, counting by 2, with no decimals
min_x = min(batches[:n_fine])
max_x = max(batches[:n_fine])
xticks = list(range(int(min_x), int(max_x)+1, 2))
ax2.set_xticks(xticks)
ax2.set_xticklabels([str(x) for x in xticks])
ax2.set_ylabel('Estimator Variance')
ax2.set_yscale('log')
ax2.legend(loc='lower right')

