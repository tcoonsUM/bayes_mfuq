#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

# import custom objects/functions
module_directory = os.path.abspath('../src/') 
sys.path.insert(0, module_directory) 
from PilotStudy import PilotStudy
import model_utils_mono
#%% comparing priors for IW 
with open('../Data/n_star_results_mono/iw_200.pickle', 'rb') as file:
    outs_orig = pickle.load(file)
    
with open('../Data/n_star_results_mono/iw_informative.pickle', 'rb') as file:
    outs_informed = pickle.load(file)


#%% set up problem 
x = torch.tensor([.5]) # this is meaningless for mono study
n_models = 4           # Number of models
w = torch.tensor([[1.],[0.1],[0.01],[0.001]]) # model costs
pilot_cost = w.sum().item()
total_budget = 200*pilot_cost
seed = 35
pilot_study = PilotStudy(model_utils=model_utils_mono, 
                         x=x, 
                         total_budget=total_budget,  
                         n_models=n_models, 
                         w=w,
                         seed=seed)
oracle_cov = model_utils_mono.sig_func().detach().numpy()

acv_best_case_var = pilot_study.compute_oracle_acv_variance(oracle_cov, total_budget, oracle_cov, w, estimator='wrdiff') 
mlmc_best_case_var = pilot_study.compute_oracle_acv_variance(oracle_cov, total_budget, oracle_cov, w, estimator='mlmc')
sf_mc_var = oracle_cov[0,0]/np.floor(total_budget)

#%% IW orig vs informative
n_test=20
true_vars_orig = np.zeros((n_test,))
true_vars_informative = np.zeros((n_test,))
for seed in range(n_test):
    sigma_p = outs_orig[seed][5]
    budget_p = outs_orig[seed][6]
    true_vars_orig[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    sigma_p = outs_informed[seed][5]
    budget_p = outs_informed[seed][6]
    true_vars_informative[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    

from matplotlib.patches import Patch
colors_three = ['#005FBE', '#DDCC77', '#BB5566']
colors_iw = ['#0071e4', '#004b98']
data = [true_vars_orig, true_vars_informative]
labels = ['Original', 'Informative']

# Create plot
fig, ax = plt.subplots(figsize=(6, 6))

# Adjust positions of boxplots to display them side-by-side at the same x-location
positions = [1 - 0.2, 1 + 0.2] 
box = ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True, showfliers=False)

# Customize medians
for median in box['medians']:
    median.set_linewidth(1)    # Match line width
    median.set_color('black')

# Set x-axis tick labels for a single location
ax.set_xticks([1])
ax.set_xticklabels([''])

# Customize box colors (optional)
for patch, color in zip(box['boxes'], colors_iw):
    patch.set_facecolor(color)

# Combine reference lines into a single continuous line across the boxplots
ax.hlines(acv_best_case_var, positions[0]-.2, positions[1]+.2, colors='gray', linestyles='--', linewidth=2, label='ACV best-case')
ax.hlines(mlmc_best_case_var, positions[0]-.2, positions[1]+.2, colors='chocolate', linestyles=(0, (3, 1, 1, 1, 1, 1)), linewidth=2, label='MLMC best-case')

# Labels and formatting
ax.set_ylabel("Variance")
ax.set_yscale('log')
ax.set_xlabel("Budget (pilot sample equivalents)")
ax.set_xticks([1])
ax.set_xticklabels(['200'])
ax.grid(True, linestyle='--', alpha=0.6)

# Create legend entries for the fill colors
# color_legend_elements = [
#     Patch(facecolor='lightgreen', edgecolor='black', label='Original'),
#     Patch(facecolor='greenyellow', edgecolor='black', label='Informative')
# ]

color_legend_elements = [
    Patch(facecolor=colors_iw[0], edgecolor='black', label='Original'),
    Patch(facecolor=colors_iw[1], edgecolor='black', label='Informative')
]

# Get existing handles and labels to combine them with new ones
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_handles.extend(color_legend_elements)
legend_labels.extend(['Original', 'Informative'])

ax.legend(legend_handles, legend_labels, loc=5)
plt.tight_layout()
plt.show()

data = [true_vars_orig, true_vars_informative]

#%% GG MVN informative vs uninformative vs original
with open('../Data/n_star_results_mono/gg_mvn_200.pickle', 'rb') as file:
    outs_orig_gg_mvn = pickle.load(file)
    
with open('../Data/n_star_results_mono/gg_mvn_informative_correct.pickle', 'rb') as file:
    outs_informed_gg_mvn = pickle.load(file)
    
with open('../Data/n_star_results_mono/gg_mvn_uninformative.pickle', 'rb') as file:
    outs_uninformed_gg_mvn = pickle.load(file)
    
#%% plotting
n_test=20
true_vars_orig_gg_mvn = np.zeros((n_test,))
true_vars_informative_gg_mvn = np.zeros((n_test,))
true_vars_uninformative_gg_mvn = np.zeros((n_test,))
for seed in range(n_test):
    sigma_p = outs_orig_gg_mvn[seed][5]
    budget_p = outs_orig_gg_mvn[seed][6]
    true_vars_orig_gg_mvn[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    sigma_p = outs_informed_gg_mvn[seed][5]
    budget_p = outs_informed_gg_mvn[seed][6]
    true_vars_informative_gg_mvn[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    sigma_p = outs_uninformed_gg_mvn[seed][5]
    budget_p = outs_uninformed_gg_mvn[seed][6]
    true_vars_uninformative_gg_mvn[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff') 

#%%    
data = [true_vars_uninformative_gg_mvn, true_vars_orig_gg_mvn, true_vars_informative_gg_mvn]
labels = ['Uninformative','Original', 'Informative']

# Create plot
fig, ax = plt.subplots(figsize=(6, 6))

# Adjust positions of boxplots to display them side-by-side
positions = [1 - 0.3, 1, 1 + 0.3]
box = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True, showfliers=True, zorder=2)

# Customize medians
for median in box['medians']:
    median.set_color('black')  # Match your desired median line color
    median.set_linewidth(1)

# Customize box colors (optional)
colors = ['#e1b6bd', '#BB5566', '#612630']  # Custom colors
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


# Combine reference lines into a single continuous line across the boxplots
ax.hlines(acv_best_case_var, positions[0] - 0.2, positions[2] + 0.2, colors='gray', linestyles='--', linewidth=2, zorder=0, label='ACV best-case')
ax.hlines(mlmc_best_case_var, positions[0] - 0.2, positions[2] + 0.2, colors='chocolate', linestyles=(0, (3, 1, 1, 1, 1, 1)), linewidth=2, zorder=1, label='MLMC best-case')

# Labels and formatting
ax.set_ylabel("Variance")
ax.set_yscale('log')
ax.set_xlabel("Budget (pilot sample equivalents)")
ax.set_xticks([1])
ax.set_xticklabels(['200'])
ax.grid(True, linestyle='--', alpha=0.6)

# Create legend entries for the fill colors
color_legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label=labels[0]),
    Patch(facecolor=colors[1], edgecolor='black', label=labels[1]),
    Patch(facecolor=colors[2], edgecolor='black', label=labels[2])
]

# Get existing handles and labels to combine them with new ones
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_handles.extend(color_legend_elements)
legend_labels.extend(labels)

ax.legend(legend_handles, legend_labels, loc='best')
plt.tight_layout()
plt.show()

