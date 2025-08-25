#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for generating "model evaluations" via multivariate normal model.
i.e. Y(d) = [f0(z,d), ... , fM(z,d)]^T ~ MVN(mu(d), Sigma(d)) 
where Sigma is oracle covariance (M x M) and mu is oracle model means (M x 1)

@author: me-tcoons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
plt.style.use('ggplot')
import corr_utils
import scipy
from pyapprox.variables.joint import IndependentMarginalsVariable
import scipy.stats as stats


# def model_eval(d, n_draws, mu_func=mean_func,Sigma_func=sig_func):
#     return np.random.multivariate_normal(mu_func(d),Sigma_func(d),size=n_draws)
    
def return_list_of_models():
    models = [lambda s: s.T**5,lambda s: s.T**4,lambda s: s.T**3,lambda s: s.T**2]
    return models

def return_z_variable():
    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    return variable

def sig_corr_func(return_corr_std=False):
    
    models = [lambda s: s.T**5,lambda s: s.T**4,lambda s: s.T**3,lambda s: s.T**2]
    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    num_true_samples = 10000
    cov_samples = np.zeros((num_true_samples,len(models)))
    cov_inputs = variable.rvs(num_true_samples)
    for ii, model in enumerate(models):
        cov_samples[:,ii] = model(cov_inputs).flatten()

    cov = torch.from_numpy(np.cov(cov_samples.T))
    corr = torch.from_numpy(np.corrcoef(cov_samples.T))
    std = torch.from_numpy(np.std(cov_samples.T,axis=1))
    
    if return_corr_std:
        return cov, corr, std
    else:
        return cov
    
def sig_func(x=None):
    
    models = [lambda s: s.T**5,lambda s: s.T**4,lambda s: s.T**3,lambda s: s.T**2]
    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    num_true_samples = 10000
    cov_samples = np.zeros((num_true_samples,len(models)))
    cov_inputs = variable.rvs(num_true_samples)
    for ii, model in enumerate(models):
        cov_samples[:,ii] = model(cov_inputs).flatten()

    cov = torch.from_numpy(np.cov(cov_samples.T))
    
    return cov

def mean_func(x=None):
    
    models = [lambda s: s.T**5,lambda s: s.T**4,lambda s: s.T**3,lambda s: s.T**2]
    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    num_true_samples = 10000
    cov_samples = np.zeros((num_true_samples,len(models)))
    cov_inputs = variable.rvs(num_true_samples)
    for ii, model in enumerate(models):
        cov_samples[:,ii] = model(cov_inputs).flatten()

    true_means = torch.from_numpy(np.mean(cov_samples,axis=0).reshape(-1,1))
    
    return true_means
    

def model_eval_seed(x, num_samples, seed=42):#sample_gmm(x, num_samples, pi, desired_mean=mean_func, desired_cov=sig_func, seed=42):

    models = [lambda s: s.T**5,lambda s: s.T**4,lambda s: s.T**3,lambda s: s.T**2]
    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    
    z_samps = variable.rvs(num_samples,random_states=[seed])
    
    samples = torch.zeros([num_samples,len(models)])
    for ii, model in enumerate(models):
        samples[:,ii] = torch.from_numpy(model(z_samps).flatten())
    
    return samples

def z_sample_seed(x, num_samples, seed=42):#sample_gmm(x, num_samples, pi, desired_mean=mean_func, desired_cov=sig_func, seed=42):

    variable = IndependentMarginalsVariable([stats.uniform(0,1)])
    z_samps = variable.rvs(num_samples,random_states=[seed])

    return z_samps

def model_eval2(d, mu_func=mean_func, Sigma_func=sig_func):
    # returns an n_draws x m tensor of MVN(mu(d), Sigma(d))
    dist = scipy.stats.multivariate_normal(mu_func(d),covariance_matrix=Sigma_func(d))
    return dist

def gamma_model(mu_vec, sigma_matrix):
    dist = scipy.stats.multivariate_normal(mu_vec, sigma_matrix)

    return dist

def gamma_model_eval(n_draws, mu_vec, sigma_matrix):
    dist = gamma_model(mu_vec, sigma_matrix)
    return dist.rvs(n_draws)


def model_dist_eval(d, n_draws, mu_func=mean_func,Sigma_func=sig_func):
    # returns an n_draws x m tensor of MVN(mu(d), Sigma(d))
    dist = scipy.stats.multivariate_normal(mu_func(d),covariance_matrix=Sigma_func(d))
    return dist.sample([n_draws])
    #return np.random.multivariate_normal(mu_func(d),Sigma_func(d),size=n_draws)

def plot_means(mu_func=mean_func, lb=-1, ub=1):
        
    res = 101
    m = 3
    x_test = torch.linspace(lb, ub, res)
    y_test = torch.zeros(m, res)
    i = 0
    for x in x_test:
        y_test[:,i] = mean_func(x)
        i += 1
        
    fig, ax = plt.subplots()
    for model in range(m):
        ax.plot(x_test, y_test[model,:],label=r'$\mu_{%s}$' %str(model))
    ax.legend()
    ax.set_title('Mean Functions')
    ax.set_xlabel(r'$\xi$')
    
    return fig,ax

def plot_hf_mean(mu_func=mean_func, lb=-1, ub=1):
        
    res = 101
    m = 3
    x_test = torch.linspace(lb, ub, res)
    y_test = torch.zeros(m, res)
    i = 0
    for x in x_test:
        y_test[:,i] = mean_func(x)
        i += 1
        
    fig, ax = plt.subplots()
    for model in range(1):
        ax.plot(x_test, y_test[model,:],label=r'$\mu_{%s}$' %str(model))
    #ax.legend()
    ax.set_title(r'$\mathbb{E}_{z}[f_{0}(z,\xi)]$')
    ax.set_xlabel(r'$\xi$')
    
    return fig,ax
        
def plot_covs(Sigma_func=sig_func, lb=-1, ub=1):
        
    res = 101
    m = 3
    n_elem = int(m*(m-1)/2)
    x_test = torch.linspace(lb, ub, res)
    y_test = torch.zeros(n_elem, res)
    ii = 0
    for x in x_test:
        y_test[:,ii] = Sigma_func(x, return_vec=True)
        ii += 1
    
    i, j = torch.triu_indices(m,m, offset=1)
    fig, ax = plt.subplots()
    for elem in range(n_elem):
        ax.plot(x_test, y_test[elem,:],label=r'$\rho_{%s,%s}$' %(str(i[elem].item()),str(j[elem].item())))
    ax.legend()
    ax.set_title('Correlation Functions')
    ax.set_xlabel(r'$\xi$')
    
    return fig,ax 

def plot_betas(Sigma_func=sig_func, lb=-1, ub=1):
        
    res = 101
    m = 3
    n_elem = int(m*(m-1)/2)
    x_test = torch.linspace(lb, ub, res)
    corrs = torch.zeros(m, m, res)
    betas = np.zeros((n_elem, res))
    ii = 0
    for x in x_test:
        corrs[:,:,ii] = Sigma_func(x)
        betas[:,ii] = corr_utils.corr_matrix_to_reals(corrs[:,:,ii].detach().numpy())
        ii += 1
    

    fig, ax = plt.subplots()
    for elem in range(n_elem):
        ax.plot(x_test, betas[elem,:],label=r'$\beta_{%s}$' %str(elem))
    ax.legend()
    ax.set_title('Beta Functions')
    ax.set_xlabel(r'$\xi$')
    
    return fig,ax 

def plot_gammas(Sigma_func=sig_func, lb=-1, ub=1):
        
    res = 101
    m = 3
    n_elem = int(m*(m-1)/2)
    x_test = torch.linspace(lb, ub, res)
    corrs = torch.zeros(m, m, res)
    betas = np.zeros((n_elem, res))
    ii = 0
    for x in x_test:
        corrs[:,:,ii] = Sigma_func(x)
        betas[:,ii] = corr_utils.GFT_forward_mapping(corrs[:,:,ii].detach().numpy())
        ii += 1
    

    fig, ax = plt.subplots()
    for elem in range(n_elem):
        ax.plot(x_test, betas[elem,:],label=r'$\gamma_{%s}$' %str(elem))
    ax.legend()
    ax.set_title('Gamma Functions')
    ax.set_xlabel(r'$\xi$')
    
    return fig,ax 



