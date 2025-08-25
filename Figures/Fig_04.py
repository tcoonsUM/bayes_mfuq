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

#%% load in data
directory = "../Data/n_star_results_mono"

# Optional: store the loaded objects in a list or dictionary
loaded_pickles = {}

# Loop over all files in the directory
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".pickle"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "rb") as f:
            try:
                loaded_pickles[filename] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
#%% set up problem 
x = torch.tensor([.5]) # this is meaningless for mono study
n_models = 4           # Number of models
w = torch.tensor([[1.],[0.1],[0.01],[0.001]]) # model costs
pilot_cost = w.sum().item()
total_budget = 200*pilot_cost
budgets = [50*pilot_cost, 100*pilot_cost, 200*pilot_cost]
seed = 35
pilot_study = PilotStudy(model_utils=model_utils_mono, 
                         x=x, 
                         total_budget=total_budget,  
                         n_models=n_models, 
                         w=w,
                         seed=seed)
oracle_cov = model_utils_mono.sig_func().detach().numpy()

acv_best_case_vars = [pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[0], oracle_cov, w, estimator='wrdiff'),\
                      pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[1], oracle_cov, w, estimator='wrdiff'),\
                      pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[2], oracle_cov, w, estimator='wrdiff')]
    
mlmc_best_case_vars = [pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[0], oracle_cov, w, estimator='mlmc'),\
                      pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[1], oracle_cov, w, estimator='mlmc'),\
                      pilot_study.compute_oracle_acv_variance(oracle_cov, budgets[2], oracle_cov, w, estimator='mlmc')]
    
sf_mc_vars = [oracle_cov[0,0]/np.floor(budgets[0]),\
              oracle_cov[0,0]/np.floor(budgets[1]),\
              oracle_cov[0,0]/np.floor(budgets[2])]
    

#%% tabulate performances
n_test=20
results_iw_200 = loaded_pickles['iw_200.pickle']
results_iw_100 = loaded_pickles['iw_100.pickle']
results_iw_50 = loaded_pickles['iw_50.pickle']
true_vars_iw_200 = np.zeros((n_test,))
true_vars_iw_100 = np.zeros((n_test,))
true_vars_iw_50 = np.zeros((n_test,))
n_iw_200 = np.zeros((n_test,))
n_iw_100 = np.zeros((n_test,))
n_iw_50 = np.zeros((n_test,))
for seed in range(n_test):
    n_iw_200[seed] = int((budgets[2] - results_iw_200[seed][6])/pilot_cost)
    sigma_p = results_iw_200[seed][5]
    budget_p = results_iw_200[seed][6]
    true_vars_iw_200[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_iw_100[seed] = int((budgets[1] - results_iw_100[seed][6])/pilot_cost)
    sigma_p = results_iw_100[seed][5]
    budget_p = results_iw_100[seed][6]
    true_vars_iw_100[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_iw_50[seed] = int((budgets[0] - results_iw_50[seed][6])/pilot_cost)
    sigma_p = results_iw_50[seed][5]
    budget_p = results_iw_50[seed][6]
    true_vars_iw_50[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')

results_gg_200 = loaded_pickles['gg_200.pickle']
results_gg_100 = loaded_pickles['gg_100.pickle']
results_gg_50 = loaded_pickles['gg_50.pickle']
true_vars_gg_200 = np.zeros((n_test,))
true_vars_gg_100 = np.zeros((n_test,))
true_vars_gg_50 = np.zeros((n_test,))
n_gg_200 = np.zeros((n_test,))
n_gg_100 = np.zeros((n_test,))
n_gg_50 = np.zeros((n_test,))
for seed in range(n_test):
    n_gg_200[seed] = int((budgets[2] - results_gg_200[seed][6])/pilot_cost)
    sigma_p = results_gg_200[seed][5]
    budget_p = results_gg_200[seed][6]
    true_vars_gg_200[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_gg_100[seed] = int((budgets[1] - results_gg_100[seed][6])/pilot_cost)
    sigma_p = results_gg_100[seed][5]
    budget_p = results_gg_100[seed][6]
    true_vars_gg_100[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_gg_50[seed] = int((budgets[0] - results_gg_50[seed][6])/pilot_cost)
    sigma_p = results_gg_50[seed][5]
    budget_p = results_gg_50[seed][6]
    true_vars_gg_50[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')

results_gg_mvn_200 = loaded_pickles['gg_mvn_200.pickle']
results_gg_mvn_100 = loaded_pickles['gg_mvn_100.pickle']
results_gg_mvn_50 = loaded_pickles['gg_mvn_50.pickle']
true_vars_gg_mvn_200 = np.zeros((n_test,))
true_vars_gg_mvn_100 = np.zeros((n_test,))
true_vars_gg_mvn_50 = np.zeros((n_test,))
n_gg_mvn_200 = np.zeros((n_test,))
n_gg_mvn_100 = np.zeros((n_test,))
n_gg_mvn_50 = np.zeros((n_test,))
for seed in range(n_test):
    n_gg_mvn_200[seed] = int((budgets[2] - results_gg_mvn_200[seed][6])/pilot_cost)
    sigma_p = results_gg_mvn_200[seed][5]
    budget_p = results_gg_mvn_200[seed][6]
    true_vars_gg_mvn_200[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_gg_mvn_100[seed] = int((budgets[1] - results_gg_mvn_100[seed][6])/pilot_cost)
    sigma_p = results_gg_mvn_100[seed][5]
    budget_p = results_gg_mvn_100[seed][6]
    true_vars_gg_mvn_100[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')
    n_gg_mvn_50[seed] = int((budgets[0] - results_gg_mvn_50[seed][6])/pilot_cost)
    sigma_p = results_gg_mvn_50[seed][5]
    budget_p = results_gg_mvn_50[seed][6]
    true_vars_gg_mvn_50[seed] = pilot_study.compute_oracle_acv_variance(sigma_p, budget_p, oracle_cov, w, estimator='wrdiff')

#%% print results
# Summary output
print("Summary of n-star Means for Each Method and Budget")
print("-" * 50)

# Means for IW
print(f"IW Budget 200 Mean: {n_iw_200.mean():.3f}")
print(f"IW Budget 100 Mean: {n_iw_100.mean():.3f}")
print(f"IW Budget  50 Mean: {n_iw_50.mean():.3f}")

# Means for GG
print(f"GG Budget 200 Mean: {n_gg_200.mean():.3f}")
print(f"GG Budget 100 Mean: {n_gg_100.mean():.3f}")
print(f"GG Budget  50 Mean: {n_gg_50.mean():.3f}")

# Means for GG_MVN
print(f"GG_MVN Budget 200 Mean: {n_gg_mvn_200.mean():.3f}")
print(f"GG_MVN Budget 100 Mean: {n_gg_mvn_100.mean():.3f}")
print(f"GG_MVN Budget  50 Mean: {n_gg_mvn_50.mean():.3f}")

print("\nSummary of True Variables for Each Method")
print("-" * 50)

# True vars for IW
print(f"IW Budget 200 True Var Mean: {true_vars_iw_200.mean():.3e}")
print(f"IW Budget 100 True Var Mean: {true_vars_iw_100.mean():.3e}")
print(f"IW Budget  50 True Var Mean: {true_vars_iw_50.mean():.3e}")

# True vars for GG
print(f"GG Budget 200 True Var Mean: {true_vars_gg_200.mean():.3e}")
print(f"GG Budget 100 True Var Mean: {true_vars_gg_100.mean():.3e}")
print(f"GG Budget  50 True Var Mean: {true_vars_gg_50.mean():.3e}")

# True vars for GG_MVN
print(f"GG_MVN Budget 200 True Var Mean: {true_vars_gg_mvn_200.mean():.3e}")
print(f"GG_MVN Budget 100 True Var Mean: {true_vars_gg_mvn_100.mean():.3e}")
print(f"GG_MVN Budget  50 True Var Mean: {true_vars_gg_mvn_50.mean():.3e}")

print("\nVariance Reduction Ratios for Each Method")
print("-" * 50)

# Variance reduction ratios for IW
var_reduction_ratio_iw_means = [
    ((oracle_cov[0, 0] / budget) / vars).mean()
    for budget, vars in zip(budgets, [true_vars_iw_50, true_vars_iw_100, true_vars_iw_200])
]
var_reduction_ratio_iw_stds = [
    ((oracle_cov[0, 0] / budget) / vars).std()
    for budget, vars in zip(budgets, [true_vars_iw_50, true_vars_iw_100, true_vars_iw_200])
]

for budget, mean, std in zip(budgets, var_reduction_ratio_iw_means, var_reduction_ratio_iw_stds):
    print(f"IW Budget {budget} Var Reduction Ratio Mean: {mean:.3f}")
    print(f"IW Budget {budget} Var Reduction Ratio Std Dev: {std:.3f}")

# Variance reduction ratios for GG
var_reduction_ratio_gg_means = [
    ((oracle_cov[0, 0] / budget) / vars).mean()
    for budget, vars in zip(budgets, [true_vars_gg_50, true_vars_gg_100, true_vars_gg_200])
]
var_reduction_ratio_gg_stds = [
    ((oracle_cov[0, 0] / budget) / vars).std()
    for budget, vars in zip(budgets, [true_vars_gg_50, true_vars_gg_100, true_vars_gg_200])
]

for budget, mean, std in zip(budgets, var_reduction_ratio_gg_means, var_reduction_ratio_gg_stds):
    print(f"GG Budget {budget} Var Reduction Ratio Mean: {mean:.3f}")
    print(f"GG Budget {budget} Var Reduction Ratio Std Dev: {std:.3f}")

# Variance reduction ratios for GG_MVN
var_reduction_ratio_gg_mvn_means = [
    ((oracle_cov[0, 0] / budget) / vars).mean()
    for budget, vars in zip(budgets, [true_vars_gg_mvn_50, true_vars_gg_mvn_100, true_vars_gg_mvn_200])
]
var_reduction_ratio_gg_mvn_stds = [
    ((oracle_cov[0, 0] / budget) / vars).std()
    for budget, vars in zip(budgets, [true_vars_gg_mvn_50, true_vars_gg_mvn_100, true_vars_gg_mvn_200])
]

for budget, mean, std in zip(budgets, var_reduction_ratio_gg_mvn_means, var_reduction_ratio_gg_mvn_stds):
    print(f"GG_MVN Budget {budget} Var Reduction Ratio Mean: {mean:.3f}")
    print(f"GG_MVN Budget {budget} Var Reduction Ratio Std Dev: {std:.3f}")
    
# Best case ACV
print(f"ACV-BEST Budget 200 Vars: {acv_best_case_vars[2]:.3e}")
print(f"ACV-BEST Budget 100 Vars: {acv_best_case_vars[1]:.3e}")
print(f"ACV-BEST Budget  50 Vars: {acv_best_case_vars[0]:.3e}")

# Best case MLMC
print(f"MLMC-BEST Budget 200 Vars: {mlmc_best_case_vars[2]:.3e}")
print(f"MLMC-BEST Budget 100 Vars: {mlmc_best_case_vars[1]:.3e}")
print(f"MLMC-BEST Budget  50 Vars: {mlmc_best_case_vars[0]:.3e}")

# Best case ACV
print(f"ACV-BEST Budget 200 VRRs: {sf_mc_vars[2]/acv_best_case_vars[2]:.3f}")
print(f"ACV-BEST Budget 100 VRRs: {sf_mc_vars[1]/acv_best_case_vars[1]:.3f}")
print(f"ACV-BEST Budget  50 VRRs: {sf_mc_vars[0]/acv_best_case_vars[0]:.3f}")

# Best case MLMC
print(f"MLMC-BEST Budget 200 VRRs: {sf_mc_vars[2]/mlmc_best_case_vars[2]:.3f}")
print(f"MLMC-BEST Budget 100 VRRs: {sf_mc_vars[1]/mlmc_best_case_vars[1]:.3f}")
print(f"MLMC-BEST Budget  50 VRRs: {sf_mc_vars[0]/mlmc_best_case_vars[0]:.3f}")

#%% combined plot
# Combine data
from matplotlib.patches import Patch
data = [
    true_vars_iw_50, true_vars_gg_50, true_vars_gg_mvn_50,
    true_vars_iw_100, true_vars_gg_100, true_vars_gg_mvn_100,
    true_vars_iw_200, true_vars_gg_200, true_vars_gg_mvn_200
]

# Define positions for boxplots
positions = [
    1, 1.2, 1.4,  # Budget 50
    2, 2.2, 2.4,  # Budget 100
    3, 3.2, 3.4   # Budget 200
]

colors_three = ['#0071e4', '#DDCC77', '#BB5566']

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Add reference lines in the background on budget group basis
for budget_index, x_center in enumerate([1.2, 2.2, 3.2]):
    ax.hlines(acv_best_case_vars[budget_index], x_center - 0.5, x_center + 0.45, colors='gray', linestyles='--', linewidth=2, label='ACV-BEST' if budget_index == 0 else None, zorder=1)
    ax.hlines(mlmc_best_case_vars[budget_index], x_center - 0.5, x_center + 0.45, colors='chocolate', linestyles=(0, (3, 1, 1, 1, 1, 1)), linewidth=2, label='MLMC-BEST' if budget_index == 0 else None, zorder=1)
    ax.hlines(sf_mc_vars[budget_index], x_center - 0.5, x_center + 0.45, colors='black', linestyles='dotted', linewidth=2, label='MC' if budget_index == 0 else None, zorder=1)

# Create boxplot with higher zorder to overlay it on the hlines
box = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True, showfliers=False, zorder=2)

# Customize medians
for median in box['medians']:
    median.set_linewidth(1)
    median.set_color('black')

# Set colors
colors = colors_three * 3  # Repeat colors for each set
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Labels and formatting
ax.set_ylabel("Variance")
ax.set_xlabel("Budget (pilot sample equivalents)")
ax.set_xticks([1.2, 2.2, 3.2])
ax.set_xticklabels(['50', '100', '200'])
ax.set_yscale('log')
ax.grid(True, linestyle='--', alpha=0.6)

# Create legend entries using patches for colors
color_legend_elements = [
    Patch(facecolor=colors_three[0], edgecolor='black', label='ADAPT-IW'),
    Patch(facecolor=colors_three[1], edgecolor='black', label='ADAPT-GG'),
    Patch(facecolor=colors_three[2], edgecolor='black', label='ADAPT-GG_MVN'),
]

# Get existing handles and labels to combine them with new ones
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_handles.extend(color_legend_elements)
legend_labels.extend(['ADAPT-IW', 'ADAPT-GG', 'ADAPT-GG_MVN'])

ax.legend(legend_handles, legend_labels, loc='upper right')
plt.tight_layout()
plt.show()

