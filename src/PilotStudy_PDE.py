#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 08:34:32 2024

@author: me-tcoons
"""

import os
# this loop helps us identify the correct user and saves 5 milliseconds ;-)
if os.environ['HOME'] == '/home/ajivani':
    user='Aniket'
else:
    user='Thomas'

import numpy as np
from numpy import linalg as la
# import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from rich.progress import track
import pickle
import sys
import copy
import gpytorch
import botorch
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from sklearn.preprocessing import StandardScaler
import contextlib 
from contextlib import nullcontext

import torch

if user=='Aniket':
    import mxmc
    from mxmc import Optimizer
    from mxmc import Estimator
    file_path = '/home/ajivani/Covuq/CSE2025'
    example_path = '/home/ajivani/Covuq/Paper01/'
    sys.path.insert(0, file_path)
    sys.path.insert(1, example_path)
    import model_utils
    # sys.path.insert(2, "/home/ajivani/Covuq/CSE2025")
    # import model_utils_mono
    import hf_model_utils
    import corr_utils
    import pilot_cov_utils
    import viz_utils as viz
    import custom_acqf
elif user=='Thomas':
    import mxmc
    from mxmc import Optimizer
    from mxmc import Estimator
    # import model_utils_mono
    import hf_model_utils
    import model_utils
    import viz_utils as viz
    import corr_utils as corr_utils
    import custom_acqf
    import pilot_cov_utils

import torch
from torch.distributions import Wishart, MultivariateNormal
import scipy.stats as stats
from alive_progress import alive_bar
import gacv as g
import scipy.optimize as opt

def calculate_cov_delta_terms(estimator, true_cov):
        k_0 = estimator._allocation.get_k0_matrix()
        k = estimator._allocation.get_k_matrix()
        cov_q_delta = k_0 * true_cov[0, 1:]
        cov_delta_delta = k * true_cov[1:, 1:]
        return cov_delta_delta, cov_q_delta
    
class PilotStudy:
    def __init__(self, model_utils, x, total_budget, n_models, w, grid, yaml_data, seed=43):
        """
        Initialize the PilotStudy class.
        
        Args:
            model_utils: Utility module for model evaluation and related functions.
            x (tensor): Input variables for the model.
            total_budget (float): Total available budget for pilot sampling.
            pilot_cost (float): Cost per pilot sample.
            n_models (int): Number of models being evaluated.
            w (numpy array or tensor): Array of model costs.
            seed (int): Random seed for reproducibility
        """
        self.model_utils = model_utils
        self.x = torch.tensor([x]) if not torch.is_tensor(x) else x
        self.total_budget = total_budget
        self.w = torch.tensor(w) if not torch.is_tensor(w) else w
        self.pilot_cost = self.w.sum().item()
        self.n_models = n_models
        self.seed = seed
        self.iw_initialized = False
        self.siw_initialized = False
        self.gamma_prior_initialized = False
        self.gamma_mvn_prior_initialized = False
        self.log_sigma_prior_initialized = False
        self.grid = grid
        self.config = yaml_data
        
    def initialize_prior(self, method='iw', nu_prior=5, 
                         S_prior = np.array([[1, 0.8, 0.6], [0.8, 1, 0.7], [0.6, 0.7, 1]]) ):
        """
        Initialize prior method and parameters.
        
        Args:
            method (char, optional): prior family. Currently either Wishart ('iw') or Shrinkage Inv Wishart ('siw') 
            nu_prior (int, optional): Prior dof for the covariance ($\nu$ for iw, $a$ for siw). Default is 5.
            S_prior (numpy array, optional): Prior scale matrix. Default is a predefined matrix.

        """
        if method=='iw':
            self.S_prior = S_prior
            self.nu_prior = nu_prior
            self.iw_initialized = True
            print("Inverse Wishart prior initialized.")
        elif method=='siw':
            self.H_prior = S_prior
            self.a_prior = nu_prior
            self.siw_initialized = True
            print("Shrinkage Inverse Wishart prior initialized.")
        else:
            print("Initialization failed. Use wis or siw priors or call initialize_gamma_prior for gamma gaussian priors.")
            
    def initialize_gamma_prior(self, mu_prior=np.array([0.,0.,0.]), sig_prior = np.array([10.,10.,10.]) ):
        """
        Initialize gamma priors: independent Gaussians on bijective parameters of corr.
        
        Args: 
            mu_prior (numpy array, optional): Prior $\gamma$ means
            sig_prior (numpy array, optional): Prior $\gamma$ standard deviations

        """
        self.mu_prior_gamma = mu_prior
        self.sig_prior_gamma = sig_prior
        self.gamma_prior_initialized = True
        
    def initialize_gamma_mvn_prior(self, mu_prior, Sig_prior):
        """
        Initialize gamma priors: MVN Gaussian on bijective parameters of corr.
        
        Args: 
            mu_prior (numpy array, optional): Prior $\gamma$ means
            Sig_prior (numpy array, optional): Prior $\gamma$ cov matrix

        """
        self.mu_prior_gamma_mvn = mu_prior
        self.Sig_prior_gamma_mvn = Sig_prior
        self.gamma_mvn_prior_initialized = True
        
    def initialize_log_sigma_prior(self, log_sig_means=np.array([0., 0., 0.]), log_sig_sds = np.array([1., 1., 1.]) ):
        """
        Initialize sigma priors: independent lognormals on stdevs.
        
        Args:
            method (char, optional): prior family. Currently either Wishart ('iw') or Shrinkage Inv Wishart ('siw') 
            log_sig_means (numpy array, optional): Prior means for $\log(\sigma)$
            log_sig_sds (numpy array, optional): Prior stdevs for $\log(\sigma)$

        """
        self.log_sigma_prior_means = log_sig_means
        self.log_sigma_prior_sds = log_sig_sds
        self.log_sigma_prior_initialized = True
        
    def gamma_updates(self, y_pilot, mu_prior=None, sig_prior=None, wishart=True, save_to_object=False, winsor=False, seed=None):
        """
        Perform Bayesian update on gamma's given y_pilot.
        
        Args:
            y_pilot (numpy array, (n_pilot x n_models)): pilot samples from model ensemble
            mu_prior (numpy array, (n_models,)): prior means on gamma gaussians
            sig_prior (numpy array, (n_models,)): prior stdevs on gamma gaussians
            wishart (bool): if True, uses Wishart dist for UQ on cov(y_pilot)
        
        Returns:
            mu_post (numpy array, (n_models,)): posterior means on gamma gaussians
            sig_post (numpy array, (n_models,)): posterior stdevs on gamma gaussians

        """
        
        # check optional inputs
        if mu_prior is None:
            mu_prior = self.mu_prior_gamma
        if sig_prior is None:
            sig_prior = self.sig_prior_gamma
        if seed is None:
            seed=self.seed
            
        n_gamma = int(self.n_models*(self.n_models-1)/2)
        n_pilot = y_pilot.shape[1]
        
        # first use y_pilot to get variance matrix of c_hat
        n_draws = 5000
        if wishart:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart(y_pilot, n_draws, return_corr_sds=True, seed=seed)
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples(y_pilot, n_draws, return_corr_sds=True, seed=seed)
        
        # propagate to gamma's
        gamma_samps_init = corr_utils.corrs_to_gammas_1d(rho_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            gam_var = gamma_samps_init.var(axis=1)
        else:
            #gamma_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=gamma_samps_init)
            #gam_var = gamma_winsorized.var(axis=1)
            # Compute 30th and 70th percentiles per row
            lower = np.percentile(gamma_samps_init, 20, axis=1)
            upper = np.percentile(gamma_samps_init, 80, axis=1)
            
            # Truncate each row to its [30th, 70th] percentile range
            gamma_truncated = np.array([
                np.clip(row, lo, hi) for row, lo, hi in zip(gamma_samps_init, lower, upper)
            ])
            
            # Compute variance per row
            gam_var = gamma_truncated.var(axis=1)
        gam_mean = gamma_samps_init.mean(axis=1)
        
        # posterior var, mean
        mu_post = np.zeros((n_gamma,))
        sig_post = np.zeros((n_gamma,))
        
        # perform conjugate Gaussian update formula 
        for m in range(n_gamma):
            var0 = sig_prior[m]**2; mu0 = mu_prior[m]
            vary = gam_var[m]; muy = gam_mean[m]
            
            var_post = 1/(1/var0 + 1/vary)
            mean_post = var_post*(mu0/var0 + muy/vary)
            
            sig_post[m] = np.sqrt(var_post)
            mu_post[m] = mean_post
            
        if save_to_object:
            self.mu_posterior_gamma = mu_post
            self.sig_posterior_gamma = sig_post
            
        return mu_post, sig_post
    
    def gamma_updates_mvn(self, y_pilot, mu_prior=None, Sig_prior=None, wishart=True, save_to_object=False, winsor=False, seed=None):
        """
        Perform Bayesian update on gamma's given y_pilot.
        
        Args:
            y_pilot (numpy array, (n_pilot x n_models)): pilot samples from model ensemble
            mu_prior (numpy array, (n_models,)): prior means on gamma gaussians
            Sig_prior (numpy array, (n_models,n_models)): prior cov matrix on gamma gaussians
            wishart (bool): if True, uses Wishart dist for UQ on cov(y_pilot)
        
        Returns:
            mu_post (numpy array, (n_models,)): posterior means on gamma gaussians
            sig_post (numpy array, (n_models,)): posterior stdevs on gamma gaussians
       
        """
        
        # check optional inputs
        if mu_prior is None:
            mu_prior = self.mu_prior_gamma_mvn
        if Sig_prior is None:
            Sig_prior = self.Sig_prior_gamma_mvn
        if seed is None:
            seed=self.seed
        
        # first use y_pilot to get variance matrix of c_hat
        n_draws = 5000
        if wishart:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart(y_pilot, n_draws, return_corr_sds=True, seed=seed)
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples(y_pilot, n_draws, return_corr_sds=True, seed=seed)
        
        # propagate to gamma's
        gamma_samps_init = corr_utils.corrs_to_gammas_1d(rho_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            gam_cov = np.cov(gamma_samps_init)
        else:
            #gamma_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=gamma_samps_init)
            #gam_var = gamma_winsorized.var(axis=1)
            # Compute 30th and 70th percentiles per row
            lower = np.percentile(gamma_samps_init, 20, axis=1)
            upper = np.percentile(gamma_samps_init, 80, axis=1)
            
            # Truncate each row to its [30th, 70th] percentile range
            gamma_truncated = np.array([
                np.clip(row, lo, hi) for row, lo, hi in zip(gamma_samps_init, lower, upper)
            ])
            
            # Compute variance per row
            gam_cov = np.cov(gamma_truncated)
        #gam_cov = np.cov(gamma_samps_init)
        gam_mean = gamma_samps_init.mean(axis=1)
        
        # perform conjugate MVN Gaussian update formula 
        cov0 = Sig_prior; mu0 = mu_prior
        covy = gam_cov; muy = gam_mean
        
        # Compute posterior covariance
        Sig_post = np.linalg.inv(np.linalg.inv(cov0) + np.linalg.inv(covy))
        
        # Compute posterior mean
        mu_post = Sig_post @ (np.linalg.inv(cov0) @ mu0 + np.linalg.inv(covy) @ muy) 
            
        if save_to_object:
            self.mu_posterior_gamma_mvn = mu_post
            self.Sig_posterior_gamma_mvn = Sig_post
            
        return mu_post, Sig_post
    
    def log_sigma_updates(self, y_pilot, log_sig_means_prior=None, log_sig_sds_prior=None, wishart=True, winsor=False, save_to_object=False, seed=None):
        """
        Perform Bayesian update on gamma's given y_pilot.
        
        Args:
            y_pilot (numpy array, (n_pilot x n_models)): pilot samples from model ensemble
            mu_prior (numpy array, (n_models,)): prior means on log-sig gaussians
            sig_prior (numpy array, (n_models,)): prior stdevs on log-sig gaussians
        
        Returns:
            mu_post (numpy array, (n_models,)): posterior means on log-sig gaussians
            sig_post (numpy array, (n_models,)): posterior stdevs on log-sig gaussians

        """
        n_pilot = y_pilot.shape[1]
        # check optional inputs
        if log_sig_means_prior is None:
            log_sig_means_prior = self.log_sigma_prior_means
        if log_sig_sds_prior is None:
            log_sig_sds_prior = self.log_sigma_prior_sds
        if seed is None:
            seed=self.seed
        
        # first use y_pilot to get variance matrix of c_hat
        n_draws = 5000
        if wishart:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart(y_pilot, n_draws, return_corr_sds=True, seed=seed)
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples(y_pilot, n_draws, return_corr_sds=True, seed=seed)

        # map to log_sigs
        log_sigs_init = np.log(sig_samps_init)
        
        # approximate gaussian likelihoods on the log_sigmas
        if not winsor:
            log_var = log_sigs_init.var(axis=1)
        else:
            log_sig_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=log_sigs_init)
            log_var = log_sig_winsorized.var(axis=1)
        log_mean = log_sigs_init.mean(axis=1)
        
        # posterior var, mean
        mu_post = np.zeros((self.n_models,))
        sig_post = np.zeros((self.n_models,))
        
        # perform conjugate Gaussian update formula 
        for m in range(self.n_models):
            var0 = log_sig_sds_prior[m]**2; mu0 = log_sig_means_prior[m]
            vary = log_var[m]; muy = log_mean[m]
            
            var_post = 1/(1/var0 + 1/vary)
            mean_post = var_post*(mu0/var0 + muy/vary)
            
            sig_post[m] = np.sqrt(var_post)
            mu_post[m] = mean_post
            
        if save_to_object:
            self.log_sigma_posterior_means = mu_post
            self.log_sigma_posterior_sds = sig_post
            
        return mu_post, sig_post
    
    def generate_gamma_gaussian_samps(self, n_samps=200, mu_post=None, sig_post=None, log_sig_means_post=None, log_sig_sds_post=None, return_corr_sds=True, seed=None):
        
        if mu_post is None:
            mu_post = self.mu_post_gamma
        if sig_post is None:
            sig_post = self.sig_post_gamma
        if log_sig_means_post is None:
            log_sig_means_post = self.log_sigma_prior_means_post
        if log_sig_sds_post is None:
            log_sig_sds_post = self.log_sigma_prior_sds_post
        if seed is None:
            seed = self.seed
            
        #sig_samps = np.zeros((self.n_models, n_samps))
        n_gamma = int(self.n_models*(self.n_models-1)/2)
        gamma_samps = np.zeros((n_gamma, n_samps))
        #corr_samps = np.zeros((self.n_models, self.n_models, n_samps))
        cov_samps = np.zeros((self.n_models, self.n_models, n_samps))
        
        
        # generate sig samps via lognormal distribution
        np.random.seed(seed)
        sig_samps = stats.lognorm.rvs(s=log_sig_sds_post,
                                      scale=np.exp(log_sig_means_post),
                                      size=(n_samps,self.n_models)).T
        # for m in range(self.n_models):
        #     np.random.seed(seed)
        #     sig_samps[m,:] = stats.lognorm.rvs(s=log_sig_sds_post[m],
        #                                        scale=np.exp(log_sig_means_post[m]),
        #                                        size=n_samps)
        # exponentiate to make sig_samps
        #sig_samps = np.exp(log_sig_samps)
        
        # generate gamma_samps
        for m in range(n_gamma):
            np.random.seed(seed)
            gamma_samps[m,:] = stats.norm.rvs(loc=mu_post[m],scale=sig_post[m],size=n_samps)
            
        # map to corr_samps
        corr_samps = corr_utils.gammas_to_corrs_1d(gamma_samps, n_models=self.n_models)
        
        # combine to make cov_samps
        for i in range(n_samps):
            sds = sig_samps[:,i]
            corr = corr_samps[:,:,i]
            cov_samps[:,:,i] = np.diag(sds) @ corr @ np.diag(sds)
            
        if return_corr_sds:
            return gamma_samps, corr_samps, sig_samps, cov_samps
        else:
            return cov_samps
        
    def generate_gamma_gaussian_samps_mvn(self, n_samps=200, mu_post=None, Sig_post=None, log_sig_means_post=None, log_sig_sds_post=None, return_corr_sds=True, seed=None):
        
        if mu_post is None:
            mu_post = self.mu_post_gamma_mvn
        if Sig_post is None:
            Sig_post = self.Sig_post_gamma_mvn
        if log_sig_means_post is None:
            log_sig_means_post = self.log_sigma_prior_means_post
        if log_sig_sds_post is None:
            log_sig_sds_post = self.log_sigma_prior_sds_post
        if seed is None:
            seed = self.seed
            
        n_gamma = int(self.n_models*(self.n_models-1)/2)
        cov_samps = np.zeros((self.n_models, self.n_models, n_samps))
        
        
        # generate sig samps via lognormal distribution
        np.random.seed(seed)
        sig_samps = stats.lognorm.rvs(s=log_sig_sds_post,
                                      scale=np.exp(log_sig_means_post),
                                      size=(n_samps,self.n_models)).T
        
        # generate gamma_samps
        gamma_samps = stats.multivariate_normal.rvs(mean=mu_post, cov=Sig_post, size=n_samps, random_state=seed).T
            
        # map to corr_samps
        corr_samps = corr_utils.gammas_to_corrs_1d(gamma_samps, n_models=self.n_models)
        
        # combine to make cov_samps
        for i in range(n_samps):
            sds = sig_samps[:,i]
            corr = corr_samps[:,:,i]
            cov_samps[:,:,i] = np.diag(sds) @ corr @ np.diag(sds)
            
        if return_corr_sds:
            return gamma_samps, corr_samps, sig_samps, cov_samps
        else:
            return cov_samps
 
    def project_gg_posterior(self, n_steps, y_pilot, 
                                     return_samples=True, project_sig_means=False,
                                     mu_prior=None, sig_prior=None, 
                                     log_sig_means_prior=None, log_sig_sds_prior=None,
                                     wishart=True, seed=None, winsor=False, hierarchical=False):
        
        # check optional inputs
        if mu_prior is None:
            mu_prior = self.mu_prior_gamma
        if sig_prior is None:
            sig_prior = self.sig_prior_gamma
        if log_sig_means_prior is None:
            log_sig_means_prior = self.log_sigma_prior_means
        if log_sig_sds_prior is None:
            log_sig_sds_prior = self.log_sigma_prior_sds
        if seed is None:
            seed = self.seed
        
        # simulate observation based on n_steps ahead using quartiles
        n_gamma = int(self.n_models*(self.n_models-1)/2)
        n_pilot = y_pilot.shape[1] + n_steps
        S_pilot = np.cov(y_pilot,bias= False)
        #print(S_pilot)
        n_draws = 500
        if wishart:
            if hierarchical:
                S_samples = self.generate_gamma_gaussian_samps(n_samps=n_draws,\
                                 mu_post=mu_prior, sig_post=sig_prior, log_sig_means_post=log_sig_means_prior, 
                                 log_sig_sds_post=log_sig_sds_prior, seed=seed)
                cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart_S_hierarchical(
                                                                        S_samples, n_pilot, return_corr_sds=True, seed=seed)
            else:
                cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart_S(
                                                                        S_pilot, n_pilot, n_draws, return_corr_sds=True, seed=seed)
          
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_projection(
                                                                    y_pilot, n_pilot, n_draws, return_corr_sds=True, seed=seed)

        # first process gamma gaussians
        # propagate to gamma's
        gamma_samps_init = corr_utils.corrs_to_gammas_1d(rho_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            gam_var = gamma_samps_init.var(axis=1)
            #print(gam_var)
        else:
            #gamma_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=gamma_samps_init)
            #gam_var = gamma_winsorized.var(axis=1)
            # Compute 30th and 70th percentiles per row
            lower = np.percentile(gamma_samps_init, 20, axis=1)
            upper = np.percentile(gamma_samps_init, 80, axis=1)
            
            # Truncate each row to its [30th, 70th] percentile range
            gamma_truncated = np.array([
                np.clip(row, lo, hi) for row, lo, hi in zip(gamma_samps_init, lower, upper)
            ])
            
            # Compute variance per row
            gam_var = gamma_truncated.var(axis=1)


        gam_mean = gamma_samps_init.mean(axis=1)
        #print(gam_mean)
        
        # posterior var, mean
        mu_post = np.zeros((n_gamma,))
        sig_post = np.zeros((n_gamma,))
        
        # perform conjugate Gaussian update formula 
        for m in range(n_gamma):
            var0 = sig_prior[m]**2; mu0 = mu_prior[m]
            vary = gam_var[m]; muy = gam_mean[m]
            
            var_post = 1/(1/var0 + 1/vary)
            mean_post = var_post*(mu0/var0 + muy/vary)
            
            sig_post[m] = np.sqrt(var_post)
            mu_post[m] = mean_post

        # now process log_sigmas
        # map to log_sigs
        log_sigs_init = np.log(sig_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            log_var = log_sigs_init.var(axis=1)
        else:
            log_sig_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=log_sigs_init)
            log_var = log_sig_winsorized.var(axis=1)
        log_mean = log_sigs_init.mean(axis=1)
        
        # posterior var, mean
        log_sig_means_post = np.zeros((self.n_models,))
        log_sig_sds_post = np.zeros((self.n_models,))
        
        # perform conjugate Gaussian update formula 
        for m in range(self.n_models):
            var0 = log_sig_sds_prior[m]**2; mu0 = log_sig_means_prior[m]
            vary = log_var[m]; muy = log_mean[m]
            
            log_sig_var_post = 1/(1/var0 + 1/vary)
            log_sig_mean_post = log_sig_var_post*(mu0/var0 + muy/vary)
            
            log_sig_sds_post[m] = np.sqrt(log_sig_var_post)
            log_sig_means_post[m] = log_sig_mean_post
        
        if not project_sig_means:
            log_sig_means_post = log_sig_means_prior
        
        if return_samples:
            return self.generate_gamma_gaussian_samps(n_samps=100, mu_post=mu_post, sig_post=sig_post, 
                                             log_sig_means_post=log_sig_means_post, log_sig_sds_post=log_sig_sds_post, 
                                             return_corr_sds=True)
        else:
            return mu_post, sig_post, log_sig_means_post, log_sig_sds_post 
        
    def project_gg_posterior_hierarchical(self, n_steps, y_pilot, 
                                     return_samples=True, project_sig_means=False,
                                     mu_prior=None, sig_prior=None, 
                                     log_sig_means_prior=None, log_sig_sds_prior=None,
                                     wishart=True, seed=None, winsor=False, posterior_tuple=None):
        
        # check optional inputs
        if mu_prior is None:
            mu_prior = self.mu_prior_gamma
        if sig_prior is None:
            sig_prior = self.sig_prior_gamma
        if log_sig_means_prior is None:
            log_sig_means_prior = self.log_sigma_prior_means
        if log_sig_sds_prior is None:
            log_sig_sds_prior = self.log_sigma_prior_sds
        if seed is None:
            seed = self.seed
        if posterior_tuple is None:
            print("please pass current posterior gam_mu, gam_sig, log_sig_mu, log_sig_sig for \Sigma | D")
            return -1
        
        # simulate observation based on n_steps ahead using quartiles
        n_gamma = int(self.n_models*(self.n_models-1)/2)
        n_pilot = y_pilot.shape[1] + n_steps
        # first use y_pilot to get variance matrix of c_hat
        n_draws = 200
        if wishart:
            S_samples = self.generate_gamma_gaussian_samps(n_samps=n_draws,\
                             mu_post=posterior_tuple[0], sig_post=posterior_tuple[1], log_sig_means_post=posterior_tuple[2], 
                             log_sig_sds_post=posterior_tuple[3], seed=seed)
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart_S_hierarchical(
                                                                    S_samples[3], n_pilot, return_corr_sds=True, seed=seed)
      
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_projection(
                                                                    y_pilot, n_pilot, n_draws, return_corr_sds=True, seed=seed)

        # first process gamma gaussians
        # propagate to gamma's
        gamma_samps_init = corr_utils.corrs_to_gammas_1d(rho_samps_init,correction=True)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            gam_var = gamma_samps_init.var(axis=1)
        else:
            #gamma_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=gamma_samps_init)
            #gam_var = gamma_winsorized.var(axis=1)
            # Compute 30th and 70th percentiles per row
            lower = np.percentile(gamma_samps_init, 20, axis=1)
            upper = np.percentile(gamma_samps_init, 80, axis=1)
            
            # Truncate each row to its [30th, 70th] percentile range
            gamma_truncated = np.array([
                np.clip(row, lo, hi) for row, lo, hi in zip(gamma_samps_init, lower, upper)
            ])
            
            # Compute variance per row
            gam_var = gamma_truncated.var(axis=1)

        gam_mean = gamma_samps_init.mean(axis=1)
        
        # posterior var, mean
        mu_post = np.zeros((n_gamma,))
        sig_post = np.zeros((n_gamma,))
        
        # perform conjugate Gaussian update formula 
        for m in range(n_gamma):
            var0 = sig_prior[m]**2; mu0 = mu_prior[m]
            vary = gam_var[m]; muy = gam_mean[m]
            
            var_post = 1/(1/var0 + 1/vary)
            mean_post = var_post*(mu0/var0 + muy/vary)
            
            sig_post[m] = np.sqrt(var_post)
            mu_post[m] = mean_post

        # now process log_sigmas
        # map to log_sigs
        log_sigs_init = np.log(sig_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        log_var = log_sigs_init.var(axis=1)
        log_mean = log_sigs_init.mean(axis=1)
        
        # posterior var, mean
        log_sig_means_post = np.zeros((self.n_models,))
        log_sig_sds_post = np.zeros((self.n_models,))
        
        # perform conjugate Gaussian update formula 
        for m in range(self.n_models):
            var0 = log_sig_sds_prior[m]**2; mu0 = log_sig_means_prior[m]
            vary = log_var[m]; muy = log_mean[m]
            
            log_sig_var_post = 1/(1/var0 + 1/vary)
            log_sig_mean_post = log_sig_var_post*(mu0/var0 + muy/vary)
            
            log_sig_sds_post[m] = np.sqrt(log_sig_var_post)
            log_sig_means_post[m] = log_sig_mean_post
        
        if not project_sig_means:
            log_sig_means_post = log_sig_means_prior
        
        if return_samples:
            return self.generate_gamma_gaussian_samps(n_samps=100, mu_post=mu_post, sig_post=sig_post, 
                                             log_sig_means_post=log_sig_means_post, log_sig_sds_post=log_sig_sds_post, 
                                             return_corr_sds=True)
        else:
            return mu_post, sig_post, log_sig_means_post, log_sig_sds_post      
 
    def project_gg_mvn_posterior(self, n_steps, y_pilot, 
                                     return_samples=True, project_sig_means=False,
                                     mu_prior=None, Sig_prior=None, 
                                     log_sig_means_prior=None, log_sig_sds_prior=None,
                                     wishart=True, winsor=False, seed=None):
        
        # check optional inputs
        if mu_prior is None:
            mu_prior = self.mu_prior_gamma_mvn
        if Sig_prior is None:
            Sig_prior = self.Sig_prior_gamma_mvn
        if log_sig_means_prior is None:
            log_sig_means_prior = self.log_sigma_prior_means
        if log_sig_sds_prior is None:
            log_sig_sds_prior = self.log_sigma_prior_sds
        if seed is None:
            seed = self.seed
        
        # simulate observation based on n_steps ahead using Wishart or sample approx. from TDixon
        n_pilot = y_pilot.shape[1] + n_steps
        S_pilot = np.cov(y_pilot,bias=True)
        # first use y_pilot to get variance matrix of c_hat
        n_draws = 5000
        if wishart:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_wishart_S(
                                                                    S_pilot, n_pilot, n_draws, return_corr_sds=True, seed=seed)
        else:
            cov_samps_init, rho_samps_init, sig_samps_init = pilot_cov_utils.generate_cov_samples_projection(
                                                                    y_pilot, n_pilot, n_draws, return_corr_sds=True, seed=seed)

        # first process gamma gaussians
        # propagate to gamma's
        gamma_samps_init = corr_utils.corrs_to_gammas_1d(rho_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            gam_cov = np.cov(gamma_samps_init)
        else:
            #gamma_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=gamma_samps_init)
            #gam_var = gamma_winsorized.var(axis=1)
            # Compute percentiles per row
            lower = np.percentile(gamma_samps_init, 20, axis=1)
            upper = np.percentile(gamma_samps_init, 80, axis=1)
            
            # Truncate each row to its percentile range
            gamma_truncated = np.array([
                np.clip(row, lo, hi) for row, lo, hi in zip(gamma_samps_init, lower, upper)
            ])
            
            # Compute covariance per row
            gam_cov = np.cov(gamma_truncated)

        #gam_cov = np.cov(gamma_samps_init)
        gam_mean = gamma_samps_init.mean(axis=1)
        
        # perform conjugate MVN Gaussian update formula 
        cov0 = Sig_prior; mu0 = mu_prior
        covy = gam_cov; muy = gam_mean
        
        # Compute posterior covariance
        Sig_post = np.linalg.inv(np.linalg.inv(cov0) + np.linalg.inv(covy))
        
        # Compute posterior mean
        mu_post = Sig_post @ (np.linalg.inv(cov0) @ mu0 + np.linalg.inv(covy) @ muy) 

        # now process log_sigmas
        # map to log_sigs
        log_sigs_init = np.log(sig_samps_init)
        
        # approximate gaussian likelihoods on the gammas
        if not winsor:
            log_var = log_sigs_init.var(axis=1)
        else:
            log_sig_winsorized = np.apply_along_axis(lambda x: stats.mstats.winsorize(x, limits=(0.3, 0.3)), axis=1, arr=log_sigs_init)
            log_var = log_sig_winsorized.var(axis=1)
        log_mean = log_sigs_init.mean(axis=1)
        
        # posterior var, mean
        log_sig_means_post = np.zeros((self.n_models,))
        log_sig_sds_post = np.zeros((self.n_models,))
        
        # perform conjugate Gaussian update formula 
        for m in range(self.n_models):
            var0 = log_sig_sds_prior[m]**2; mu0 = log_sig_means_prior[m]
            vary = log_var[m]; muy = log_mean[m]
            
            log_sig_var_post = 1/(1/var0 + 1/vary)
            log_sig_mean_post = log_sig_var_post*(mu0/var0 + muy/vary)
            
            log_sig_sds_post[m] = np.sqrt(log_sig_var_post)
            log_sig_means_post[m] = log_sig_mean_post
        
        if not project_sig_means:
            log_sig_means_post = log_sig_means_prior
        
        if return_samples:
            return self.generate_gamma_gaussian_samps_mvn(n_samps=100, mu_post=mu_post, Sig_post=Sig_post, 
                                             log_sig_means_post=log_sig_means_post, log_sig_sds_post=log_sig_sds_post, 
                                             return_corr_sds=True)
        else:
            return mu_post, Sig_post, log_sig_means_post, log_sig_sds_post
        
        
    def calculate_cov_delta_terms(self, estimator, true_cov):
        """
        Calculate the covariance delta terms needed for loss computation.

        Args:
            estimator: Estimator object used for obtaining allocation matrices.
            true_cov (numpy array): The true covariance matrix.

        Returns:
            tuple: Covariance delta matrices (cov_delta_delta, cov_q_delta).
        """
        k_0 = estimator._allocation.get_k0_matrix()
        k = estimator._allocation.get_k_matrix()
        cov_q_delta = k_0 * true_cov[0, 1:]
        cov_delta_delta = k * true_cov[1:, 1:]
        return cov_delta_delta, cov_q_delta

    def compute_oracle_acv_variance(self, cov, b, oracle_cov, w=None, estimator='wrdiff', return_projection=False):
        
        if w is None:
            w = self.w.detach().numpy().flatten()      
        else:
            w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
            
        variance_results_pilot = dict()
        sample_allocation_results_pilot = dict()
        mxmc_optimizer = Optimizer(w, cov)
        
        for algorithm in [estimator]:
            opt_result = mxmc_optimizer.optimize(algorithm, b)
            variance_results_pilot[algorithm] = opt_result.variance
            sample_allocation_results_pilot[algorithm] = opt_result.allocation
            
        proj_var = opt_result.variance
        best_method_pilot = min(variance_results_pilot, key=variance_results_pilot.get)
        sample_allocation_pilot = sample_allocation_results_pilot[best_method_pilot]
        estimator = Estimator(sample_allocation_pilot, cov)
        
        cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator, oracle_cov)
        alpha = estimator._calculate_alpha()
        var_sf = oracle_cov[0, 0] / sample_allocation_pilot.get_number_of_samples_per_model()[0]
        true_var = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
        
        if return_projection:
            return true_var, proj_var
        else:
            return true_var

    def compute_expected_cost_loss(self, c_hat, cov_samps, b_acv, b_full, w, return_vec=False):
        """
        Compute the expected total loss based on posterior covariance samples.
    
        Args:
            c_hat (torch.Tensor): Estimated covariance matrix.
            cov_samps (torch.Tensor): Covariance samples.
            b_acv (float): Budget allocated for ACV estimation.
            b_full (float): Total available budget.
            w (torch.Tensor): Model costs.
            return_vec (bool, optional): If True, returns full vector of cost losses.
    
        Returns:
            float: Mean of expected total losses.
            Optional (if return_vec is True): Tuple containing the mean loss, 
                                              individual losses, variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        # Local numpy array conversion
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]
    
        estimator_list, sample_allocation_list = [], []
        losses, var_opts, var_b_acvs = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, cov)
            for algorithm in ["wrdiff"]: # fix this later!
                opt_result = mxmc_optimizer.optimize(algorithm, b_full)
                variance_results[algorithm] = opt_result.variance
                sample_allocation_results[algorithm] = opt_result.allocation
            best_method_pilot = min(variance_results, key=variance_results.get)
            sample_allocation = sample_allocation_results[best_method_pilot]
            estimator_list.append(Estimator(sample_allocation, cov))
            sample_allocation_list.append(sample_allocation)
            
            variance_results_b_acv = dict()
            sample_allocation_results_b_acv = dict()
            mxmc_optimizer_b_acv = Optimizer(w, cov)
            for algorithm in ["wrdiff"]: # fix this later!
                opt_result = mxmc_optimizer_b_acv.optimize(algorithm, b_acv)
                variance_results[algorithm] = opt_result.variance
                sample_allocation_results[algorithm] = opt_result.allocation
            best_method_pilot = min(variance_results_b_acv, key=variance_results_b_acv.get)
            sample_allocation_b_acv = sample_allocation_results_b_acv[best_method_pilot]
            estimator_b_acv = Estimator(sample_allocation_b_acv, cov)
    
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_b_acv, cov_samps[:, :, samp])
            alpha = estimator_b_acv._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_b_acv.get_number_of_samples_per_model()[0]
            var_b_acv = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
            
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_list[samp], cov_samps[:, :, samp])
            alpha = estimator_list[samp]._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_list[samp].get_number_of_samples_per_model()[0]
            var_opt = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
    
            losses[samp] = var_b_acv - var_opt
            var_b_acvs[samp] = var_b_acv
            var_opts[samp] = var_opt
    
        #print("EI: "+str(losses.mean())+", Var[EI]: "+str(losses.var()/n_samps))
        return (losses.mean(), losses, var_b_acvs, var_opts) if return_vec else losses.mean()
    
    def compute_expected_accuracy_and_cost_losses(self, c_hat, cov_samps, b_acv, b_full, w, return_vec=False):
        """
        Compute the expected total loss based on posterior covariance samples.
    
        Args:
            c_hat (torch.Tensor): Estimated covariance matrix.
            cov_samps (torch.Tensor): Covariance samples.
            b_acv (float): Budget allocated for ACV estimation.
            w (torch.Tensor): Model costs.
            return_vec (bool, optional): If True, returns full vector of cost losses.
    
        Returns:
            float: Mean of expected total losses.
            Optional (if return_vec is True): Tuple containing the mean loss, 
                                              individual losses, variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        # Local numpy array conversion
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]
    
        # find estimator with b_acv and c_hat (our estimator)
        variance_results_pilot = dict()
        sample_allocation_results_pilot = dict()
        mxmc_optimizer = Optimizer(w, c_hat)
        algorithms = list(Optimizer.get_algorithm_names())
        for algorithm in ["wrdiff"]:#algorithms: # fix this later!!
            if algorithm in ('', 'mlmc'):
                variance_results_pilot[algorithm] = 999999.
            else:
                opt_result = mxmc_optimizer.optimize(algorithm, b_acv)
                variance_results_pilot[algorithm] = opt_result.variance
                sample_allocation_results_pilot[algorithm] = opt_result.allocation    
        best_method_pilot = min(variance_results_pilot, key=variance_results_pilot.get)
        sample_allocation_pilot = sample_allocation_results_pilot[best_method_pilot]
        estimator_c_hat = Estimator(sample_allocation_pilot, c_hat)
        #print("best_method_pilot: "+best_method_pilot)
    
        estimator_list, sample_allocation_list = [], []
        acc_losses, cost_losses, var_opts, var_b_acvs, var_c_hats = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,)),  np.zeros((n_samps,)),  np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            
            # first find best-case estimator
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, cov)
            for algorithm in ["wrdiff"]: # fix this later!
                if algorithm != best_method_pilot:
                    variance_results[algorithm] = 999999.
                else:
                    opt_result = mxmc_optimizer.optimize(algorithm, b_full) # only difference!
                    variance_results[algorithm] = opt_result.variance
                    sample_allocation_results[algorithm] = opt_result.allocation
            best_method_pilot = min(variance_results, key=variance_results.get)
            sample_allocation = sample_allocation_results[best_method_pilot]
            estimator_list.append(Estimator(sample_allocation, cov))
            sample_allocation_list.append(sample_allocation)
            
            # next find the estimator associated with b_acv and oracle_cov
            variance_results_b_acv = dict()
            sample_allocation_results_b_acv = dict()
            mxmc_optimizer_b_acv = Optimizer(w, cov)
            for algorithm in ["wrdiff"]: # fix this later!
                opt_result = mxmc_optimizer_b_acv.optimize(algorithm, b_acv)
                variance_results_b_acv[algorithm] = opt_result.variance
                sample_allocation_results_b_acv[algorithm] = opt_result.allocation
            best_method_pilot = min(variance_results_b_acv, key=variance_results_b_acv.get)
            sample_allocation_b_acv = sample_allocation_results_b_acv[best_method_pilot]
            estimator_b_acv = Estimator(sample_allocation_b_acv, cov)
    
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_b_acv, cov_samps[:, :, samp])
            alpha = estimator_b_acv._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_b_acv.get_number_of_samples_per_model()[0]
            var_b_acv = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
    
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_c_hat, cov)
            alpha = estimator_c_hat._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_pilot.get_number_of_samples_per_model()[0]
            var_c_hat = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
            
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_list[samp], cov)
            alpha = estimator_list[samp]._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_list[samp].get_number_of_samples_per_model()[0]
            var_opt = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
    
            acc_losses[samp] = var_c_hat - var_b_acv
            if acc_losses[samp]<0:
                acc_losses[samp] = 0.
            cost_losses[samp] = var_b_acv - var_opt
            if cost_losses[samp]<0:
                cost_losses[samp] = 0.
            var_c_hats[samp] = var_c_hat
            var_b_acvs[samp] = var_b_acv
            var_opts[samp] = var_opt
    
        #print("EI: "+str(losses.mean())+", Var[EI]: "+str(losses.var()/n_samps))
        return (acc_losses.mean(), acc_losses, cost_losses.mean(), cost_losses, var_c_hats, var_b_acvs, var_opts) \
            if return_vec else (acc_losses.mean(), cost_losses.mean())
    
    def compute_expected_accuracy_loss(self, c_hat, cov_samps, b_acv, w, return_vec=False):
        
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]
    
        variance_results_pilot = dict()
        sample_allocation_results_pilot = dict()
        mxmc_optimizer = Optimizer(w, c_hat)
        algorithms = list(Optimizer.get_algorithm_names())
        
        for algorithm in ["wrdiff"]:
            opt_result = mxmc_optimizer.optimize(algorithm, b_acv)
            variance_results_pilot[algorithm] = opt_result.variance
            sample_allocation_results_pilot[algorithm] = opt_result.allocation
                
        best_method_pilot = min(variance_results_pilot, key=variance_results_pilot.get)
        sample_allocation_pilot = sample_allocation_results_pilot[best_method_pilot]
        estimator_c_hat = Estimator(sample_allocation_pilot, c_hat)
    
        estimator_list, sample_allocation_list = [], []
        losses, var_opts, var_c_hats = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, cov)
    
            for algorithm in ["wrdiff"]: 
                opt_result = mxmc_optimizer.optimize(algorithm, b_acv) # only difference!
                variance_results[algorithm] = opt_result.variance
                sample_allocation_results[algorithm] = opt_result.allocation
    
            best_method_pilot = min(variance_results, key=variance_results.get)
            sample_allocation = sample_allocation_results[best_method_pilot]
            estimator_list.append(Estimator(sample_allocation, cov))
            sample_allocation_list.append(sample_allocation)
    
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_c_hat, cov)
            alpha = estimator_c_hat._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_pilot.get_number_of_samples_per_model()[0]
            var_c_hat = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
            
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_list[samp], cov)
            alpha = estimator_list[samp]._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_list[samp].get_number_of_samples_per_model()[0]
            var_opt = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
    
            losses[samp] = var_c_hat - var_opt
            if losses[samp]<0:
                losses[samp] = 0.
            var_c_hats[samp] = var_c_hat
            var_opts[samp] = var_opt
        
        if return_vec:
            return (losses.mean(), losses, var_c_hats, var_opts) 
        else: 
            return losses.mean() 

    def compute_expected_total_loss(self, c_hat, cov_samps, b_acv, b_full, w, return_vec=False):
        """
        Compute the expected total loss based on posterior covariance samples.
    
        Args:
            c_hat (torch.Tensor): Estimated covariance matrix.
            cov_samps (torch.Tensor): Covariance samples.
            b_acv (float): Budget allocated for ACV estimation.
            b_full (float): Total available budget.
            w (torch.Tensor): Model costs.
            return_vec (bool, optional): If True, returns full vector of cost losses.
    
        Returns:
            float: Mean of expected total losses.
            Optional (if return_vec is True): Tuple containing the mean loss, 
                                              individual losses, variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        # Local numpy array conversion
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]
    
        variance_results_pilot = dict()
        sample_allocation_results_pilot = dict()
        mxmc_optimizer = Optimizer(w, c_hat)
        algorithms = list(Optimizer.get_algorithm_names())
        
        for algorithm in ["wrdiff"]:#algorithms: # fix this later!!
            if algorithm in ('', 'mlmc'):
                variance_results_pilot[algorithm] = 999999.
            else:
                opt_result = mxmc_optimizer.optimize(algorithm, b_acv)
                variance_results_pilot[algorithm] = opt_result.variance
                sample_allocation_results_pilot[algorithm] = opt_result.allocation
                
        best_method_pilot = min(variance_results_pilot, key=variance_results_pilot.get)
        sample_allocation_pilot = sample_allocation_results_pilot[best_method_pilot]
        estimator_c_hat = Estimator(sample_allocation_pilot, c_hat)
        #print("best_method_pilot: "+best_method_pilot)
    
        estimator_list, sample_allocation_list = [], []
        losses, var_opts, var_c_hats = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, cov)
    
            for algorithm in ["wrdiff"]: # fix this later!
                if algorithm != best_method_pilot:
                    variance_results[algorithm] = 999999.
                else:
                    opt_result = mxmc_optimizer.optimize(algorithm, b_full)
                    variance_results[algorithm] = opt_result.variance
                    sample_allocation_results[algorithm] = opt_result.allocation
    
            best_method_pilot = min(variance_results, key=variance_results.get)
            sample_allocation = sample_allocation_results[best_method_pilot]
            estimator_list.append(Estimator(sample_allocation, cov))
            sample_allocation_list.append(sample_allocation)
    
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_c_hat, cov)
            alpha = estimator_c_hat._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_pilot.get_number_of_samples_per_model()[0]
            var_c_hat = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
            
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_list[samp], cov)
            alpha = estimator_list[samp]._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_list[samp].get_number_of_samples_per_model()[0]
            var_opt = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
    
            losses[samp] = var_c_hat - var_opt
            # losses[samp] = var_c_hat - var_opt
            if losses[samp]<0:
                losses[samp] = 0.
            var_c_hats[samp] = var_c_hat
            var_opts[samp] = var_opt
    
        #print("EI: "+str(losses.mean())+", Var[EI]: "+str(losses.var()/n_samps))
        return (losses.mean(), losses, var_c_hats, var_opts) if return_vec else losses.mean()

    def compute_expected_total_loss_vrr(self, c_hat, cov_samps, b_acv, b_full, w, return_vec=False):
        """
        Compute the expected total loss based on posterior covariance samples.
        VRR version!

        Args:
            c_hat (torch.Tensor): Estimated covariance matrix.
            cov_samps (torch.Tensor): Covariance samples.
            b_acv (float): Budget allocated for ACV estimation.
            b_full (float): Total available budget.
            w (torch.Tensor): Model costs.
            return_vec (bool, optional): If True, returns full vector of cost losses.

        Returns:
            float: Mean of expected total losses.
            Optional (if return_vec is True): Tuple containing the mean loss, 
                                              individual losses, variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        # Local numpy array conversion
        print("VRR Loss Version")
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]

        variance_results_pilot = dict()
        sample_allocation_results_pilot = dict()
        mxmc_optimizer = Optimizer(w, c_hat)
        algorithms = list(Optimizer.get_algorithm_names())
        
        for algorithm in ["wrdiff"]:#algorithms: # fix this later!!
            if algorithm in ('', 'mlmc'):
                variance_results_pilot[algorithm] = 999999.
            else:
                opt_result = mxmc_optimizer.optimize(algorithm, b_acv)
                variance_results_pilot[algorithm] = opt_result.variance
                sample_allocation_results_pilot[algorithm] = opt_result.allocation
                
        best_method_pilot = min(variance_results_pilot, key=variance_results_pilot.get)
        sample_allocation_pilot = sample_allocation_results_pilot[best_method_pilot]
        estimator_c_hat = Estimator(sample_allocation_pilot, c_hat)
        #print("best_method_pilot: "+best_method_pilot)

        estimator_list, sample_allocation_list = [], []
        losses, var_opts, var_c_hats = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, cov)

            for algorithm in ["wrdiff"]: # fix this later!
                if algorithm != best_method_pilot:
                    variance_results[algorithm] = 999999.
                else:
                    opt_result = mxmc_optimizer.optimize(algorithm, b_full)
                    variance_results[algorithm] = opt_result.variance
                    sample_allocation_results[algorithm] = opt_result.allocation

            best_method_pilot = min(variance_results, key=variance_results.get)
            sample_allocation = sample_allocation_results[best_method_pilot]
            estimator_list.append(Estimator(sample_allocation, cov))
            sample_allocation_list.append(sample_allocation)

            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_c_hat, cov)
            alpha = estimator_c_hat._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_pilot.get_number_of_samples_per_model()[0]
            var_c_hat = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta
            
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_list[samp], cov)
            alpha = estimator_list[samp]._calculate_alpha()
            var_sf = cov_samps[:, :, samp][0, 0] / sample_allocation_list[samp].get_number_of_samples_per_model()[0]
            var_opt = var_sf + alpha @ cov_delta_delta @ alpha + 2 * alpha @ cov_q_delta

            n_sf_mc = np.floor(b_full)
            var_sf_mc = cov_samps[:, :, samp][0, 0]/n_sf_mc
            losses[samp] = var_sf_mc/var_opt - var_sf_mc/var_c_hat
            if losses[samp]<0:
                losses[samp] = 0.
            var_c_hats[samp] = var_sf_mc/var_c_hat
            var_opts[samp] = var_sf_mc/var_opt

        #print("EI: "+str(losses.mean())+", Var[EI]: "+str(losses.var()/n_samps))
        return (losses.mean(), losses, var_c_hats, var_opts) if return_vec else losses.mean()
    
    def compute_expected_total_loss_gacv(self, c_hat, cov_samps, b_acv, b_full, w, return_vec=False):
        """
        Compute the expected total loss based on posterior covariance samples.

        Args:
            c_hat (torch.Tensor): Estimated covariance matrix.
            cov_samps (torch.Tensor): Covariance samples.
            b_acv (float): Budget allocated for ACV estimation.
            b_full (float): Total available budget.
            w (torch.Tensor): Model costs.
            return_vec (bool, optional): If True, returns full vector of cost losses.

        Returns:
            float: Mean of expected total losses.
            Optional (if return_vec is True): Tuple containing the mean loss, 
                                              individual losses, variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        models = self.model_utils.return_list_of_models()
        # variable = self.model_utils.return_variable()
        est = g.GACV(models)
        # Local numpy array conversion
        w = w.detach().numpy().flatten() if torch.is_tensor(w) else w
        c_hat = c_hat.detach().numpy() if torch.is_tensor(c_hat) else c_hat
        cost_per_group = [np.sum(w[models]) for models in est.models_per_group]
        cov_samps = cov_samps.detach().numpy() if torch.is_tensor(cov_samps) else cov_samps
        n_samps = cov_samps.shape[2]
        
        # first find "suboptimal" gacv for c_hat (sigma_prime) and b_acv
        def objective(params):
            opt_beta = est.find_opt_beta(params,c_hat)
            obj = np.log(est.find_estimator_variance(params,c_hat,opt_beta))
            return obj
        init_opt_var = np.ones((est.n_sample_groups))
        cons = ({'type':'ineq','fun':lambda params: b_acv - np.dot(cost_per_group, params[:est.n_sample_groups])},
                {'type':'ineq','fun':lambda params: params[:est.n_sample_groups] - 1})

        options = {'maxiter':250}
        res = opt.minimize(objective, init_opt_var, constraints=cons, options=options)
        if res.success == False:
            print(res)
        betas_c_hat = est.find_opt_beta(res.x,c_hat)
        allocation_c_hat = res.x
            
        # looping over posterior samples
        losses, var_opts, var_c_hats = np.zeros((n_samps,)), np.zeros((n_samps,)), np.zeros((n_samps,))
        for samp in range(n_samps):
            cov = cov_samps[:, :, samp]
            
            # first constructing gacv est for each sample and b_full
            est_samp = g.GACV(models)
            cons_samp = ({'type':'ineq','fun':lambda params_samp: b_full - np.dot(cost_per_group, params_samp)},
                         {'type':'ineq','fun':lambda params_samp: params_samp[:est_samp.n_sample_groups] - 1})
            def objective_samp(params):
                opt_beta = est.find_opt_beta(params,cov)
                obj = np.log(est.find_estimator_variance(params,cov,opt_beta))
                return obj
            res_samp = opt.minimize(objective, res.x, constraints=cons_samp, options=options)
            if res.success == False:
                print(res_samp)
            betas_samp = est_samp.find_opt_beta(res_samp.x,cov)
            allocation_samp = res_samp.x
                
            # compute variances   
            var_c_hats[samp] = est.find_estimator_variance(allocation_c_hat, cov, betas_c_hat)
            var_opts[samp] = est_samp.find_estimator_variance(allocation_samp, cov, betas_samp)
            losses[samp] = var_c_hats[samp] - var_opts[samp]

        #print("EI: "+str(losses.mean())+", Var[EI]: "+str(losses.var()/n_samps))
        return (losses.mean(), losses, var_c_hats, var_opts) if return_vec else losses.mean()

    def find_n_star_gg_steps(self, x, k=2, n_steps=5, start=None, stop=50, n_mc=200, seed=None, y_pilot_precomputed=None, sigma_prime='mean', estimator='acv', estimate=False, winsor=False, hier=False, precomputed=False):
        """
        Run gamma gaussian based pilot sampling termination algorithm.

        Args:
            x (tensor): Input variables for the model.
            k (int, optional): step size for projected posterior. Default is 2.
            n_steps (int, optional): number of pilot sample steps to project posteriors. Default is 5.
            start (int, optional): Minimum number of iterations. Default is self.n_models + 1.
            stop (int, optional): Maximum number of iterations. Default is 50.
            n_mc (int, optional): Number of MC samples for loss estimation. Default is 200.
            seed (int, optional): Random seed for reproducibility. Default is self.seed.
            sigma_prime (string, optional): Choice of point estimate for \Sigma. Default is posterior mean.
        """
        if start is None:
            start = self.n_models + 1
        if seed is None:
            seed = self.seed
        if self.gamma_prior_initialized is False:
            print("Please initialize a gamma gaussian prior first!")
            return 0
        
        l_list, l_k_list, losses_list, losses_k_list, budget_list = [], [], [], [], []
        ii = start
        
        models = self.model_utils.return_list_of_models(self.config["checkpoint_dir"], self.config)


        while ii<stop:
            pilot_budget = ii * self.pilot_cost
            budget = self.total_budget - pilot_budget
            print("Running projected loss loop for batch size "+str(ii)+" and ACV budget "+str(budget)[:5])

            if precomputed and y_pilot_precomputed is not None:
                y_pilot = y_pilot_precomputed[:ii, :].T
            else:
                _, _, _, _, y_pilot = self.model_utils.model_eval_seed(ii, self.grid, models, self.config, seed=seed)
                y_pilot = y_pilot.detach().numpy().T
            c_hat = np.cov(y_pilot)#torch.cov(y_pilot, correction=1)
            
            # compute and sample from current posterior p(Sigma|y_pilot)
            gam_post = self.gamma_updates(y_pilot, winsor=winsor, seed=seed)
            log_sig_post = self.log_sigma_updates(y_pilot, winsor=winsor, seed=seed)
            gam_samps, corr_samps, sig_samps, cov_samps = self.generate_gamma_gaussian_samps(n_mc, mu_post=gam_post[0],
            sig_post=gam_post[1],
            log_sig_means_post=log_sig_post[0],
            log_sig_sds_post=log_sig_post[1],
            seed=seed)
            
            # choose Sigma prime
            if sigma_prime == 'mean':
                sigma_p = cov_samps.mean(axis=2)
            elif sigma_prime == 'median':
                sigma_p = np.median(cov_samps,axis=2)
            elif sigma_prime == 'c_hat':
                sigma_p = c_hat.cpu().detach().numpy()
            elif sigma_prime == 'analytical_mean':
                corr_p = corr_utils.GFT_inverse_mapping(gam_post[0])[0]
                sds_p = np.exp(log_sig_post[0])
                sigma_p = np.diag(sds_p) @ corr_p @ np.diag(sds_p)
            else:
                print("Please select a valid sigma_prime method!")
                return 0

            # checking current expected loss
            if estimator == 'gacv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss_gacv(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            elif estimator == 'acv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            
            # projecting next posterior after batch_size additional samples
            l_k_steps = []
            losses_k_steps = []
            for j in range(n_steps):
                k_step = k * (j + 1)
                budget_k = budget - k_step * self.pilot_cost
                if not winsor:
                    if hier:
                        gam_mean_proj, gam_std_proj, log_sig_mean_proj, log_sig_std_proj = self.project_gg_posterior_hierarchical(k_step, y_pilot, return_samples=False, project_sig_means=True, wishart=True, seed=seed,posterior_tuple=(gam_post[0],
                        gam_post[1],log_sig_post[0],log_sig_post[1]))
                    else:
                        gam_mean_proj, gam_std_proj, log_sig_mean_proj, log_sig_std_proj = self.project_gg_posterior(k_step, y_pilot, return_samples=False, project_sig_means=True, wishart=True, seed=seed)
                else:
                    gam_mean_proj, gam_std_proj, log_sig_mean_proj, log_sig_std_proj = self.project_gg_posterior(k_step, y_pilot, return_samples=False, project_sig_means=True, wishart=True, seed=seed, winsor=True)
                # since we are not projecting log_sig_means, use *current* posterior log sig mean!!
                # actually, we want to equate the mean of the exp(log_sigs), 
                #   as in E[exp(log_sigs)]= exp(mu+(sig**2)/2),
                #   which gives constant E[exp(log_sigs)] via mu+(sig**2)/2 = mu_proj+(sig_proj**2)/2
                log_sig_means_post_proj_new_formula = log_sig_post[0] + (log_sig_post[1]**2 - log_sig_std_proj**2)/2
                # print(log_sig_means_post_proj_new_formula)
                # print(np.exp(log_sig_post[0]+log_sig_post[1]**2/2)**2)
                # print(np.exp(log_sig_means_post_proj_new_formula+log_sig_std_proj[1]**2/2)**2)
                # print(gam_post[0])
                # print(gam_mean_proj)
                # print(gam_post[1])
                # print(gam_std_proj)
                # print(log_sig_post[0])
                # print(log_sig_mean_proj)
                # print(log_sig_post[1])
                # print(log_sig_std_proj)
                gam_samps_k, corr_samps_k, sig_samps_k, cov_samps_k = self.generate_gamma_gaussian_samps(n_mc, mu_post=gam_mean_proj,
                sig_post=gam_std_proj,
                log_sig_means_post=log_sig_post[0],#log_sig_means_post_proj_new_formula,#log_sig_mean_proj,#log_sig_means_post_proj_new_formula,
                log_sig_sds_post=log_sig_std_proj,
                seed=seed)  
            
                # choose Sigma prime
                if sigma_prime == 'mean':
                    sigma_p_k = corr_utils.nearestPD(cov_samps_k.mean(axis=2))
                elif sigma_prime == 'median':
                    sigma_p_k = np.median(cov_samps_k,axis=2)
                elif sigma_prime == 'c_hat':
                    sigma_p_k = c_hat.cpu().detach().numpy()
                elif sigma_prime == 'analytical_mean':
                    corr_p_k = corr_utils.GFT_inverse_mapping(gam_mean_proj)[0]
                    sds_p_k = np.exp(log_sig_means_post_proj_new_formula)
                    sigma_p_k = np.diag(sds_p_k) @ corr_p_k @ np.diag(sds_p_k)
                
                # print( sigma_p)
                # print( sigma_p_k)
            
                if estimator == 'gacv':
                    l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss_gacv(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)
                elif estimator == 'acv':
                    l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)

                l_k_steps.append(l_k)
                losses_k_steps.append(losses_k)
            
            # record each step's data
            l_list.append(l)
            losses_list.append(losses)
            # l_k_list.append(l_k)
            # losses_k_list.append(losses_k)
            l_k_list.append(l_k_steps)
            losses_k_list.append(losses_k_steps)
            budget_list.append(budget)
            print(l,l_k_steps)
            # print(f"Current losses: {l:.5e}; projected losses: {l_k:.5e}")
            #print("Current losses: "+str(l)[:7]+"; projected losses: "+str(l_k)[:7])
            
            # stopping criterion
            # if l<l_k:
            if all(l < l_step for l_step in l_k_steps):
                print("Termination met at iteration "+str(ii))
                if not estimate:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
                else:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k,seed=seed)
            
            ii+=k
        print("No termination criterion met, terminating pilot sampling at end condition iteration "+str(ii))
        if not estimate:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
        else:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k, seed=seed)

    def find_n_star_gg_mvn(self, x, batch_size=1, start=None, stop=50, n_mc=200, seed=None, y_pilot_precomputed=None, sigma_prime='mean', estimator='acv', winsor=False, estimate=False, precomputed=False):
        """
        Run gamma gaussian MVN based pilot sampling termination algorithm.
    
        Args:
            x (tensor): Input variables for the model.
            batch_size (int, optional): Size of each batch for sampling. Default is 1.
            start (int, optional): Minimum number of iterations. Default is self.n_models + 1.
            stop (int, optional): Maximum number of iterations. Default is 50.
            n_mc (int, optional): Number of MC samples for loss estimation. Default is 200.
            seed (int, optional): Random seed for reproducibility. Default is self.seed.
            sigma_prime (string, optional): Choice of point estimate for \Sigma. Default is posterior mean.
        """
        if start is None:
            start = self.n_models + 1
        if seed is None:
            seed = self.seed
        if self.gamma_mvn_prior_initialized is False:
            print("Please initialize a gamma gaussian MVN prior first!")
            return 0
        
        l_list, l_k_list, losses_list, losses_k_list, budget_list = [], [], [], [], []
        ii = start
        
        models = self.model_utils.return_list_of_models(self.config["checkpoint_dir"], self.config)
    
        while ii<stop:
            pilot_budget = ii * self.pilot_cost
            budget = self.total_budget - pilot_budget
            print("Running projected loss loop for batch size "+str(ii)+" and ACV budget "+str(budget)[:5])

            if precomputed and y_pilot_precomputed is not None:
                y_pilot = torch.Tensor(y_pilot_precomputed[:ii, :].T)
            else:
                _, _, _, _, y_pilot = self.model_utils.model_eval_seed(ii, self.grid, models, self.config, seed=seed)
                y_pilot = y_pilot.detach().T
            c_hat = torch.cov(y_pilot, correction=1)
            
            # compute and sample from current posterior p(Sigma|y_pilot)
            gam_post = self.gamma_updates_mvn(y_pilot, winsor=winsor, seed=seed)
            log_sig_post = self.log_sigma_updates(y_pilot, winsor=winsor, seed=seed)
            gam_samps, corr_samps, sig_samps, cov_samps = self.generate_gamma_gaussian_samps_mvn(n_mc, mu_post=gam_post[0],
                                            Sig_post=gam_post[1],
                                            log_sig_means_post=log_sig_post[0],
                                            log_sig_sds_post=log_sig_post[1],
                                            seed=seed)
            
            # choose Sigma prime
            if sigma_prime == 'mean':
                sigma_p = cov_samps.mean(axis=2)
            elif sigma_prime == 'median':
                sigma_p = np.median(cov_samps,axis=2)
            elif sigma_prime == 'c_hat':
                sigma_p = c_hat.cpu().detach().numpy()
            elif sigma_prime == 'analytical_mean':
                corr_p = corr_utils.GFT_inverse_mapping(gam_post[0])[0]
                sds_p = np.exp(log_sig_post[0])
                sigma_p = np.diag(sds_p) @ corr_p @ np.diag(sds_p)
            else:
                print("Please select a valid sigma_prime method!")
                return 0
    
            # checking current expected loss
            if estimator == 'gacv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss_gacv(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            elif estimator == 'acv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            
            # projecting next posterior after batch_size additional samples
            budget_k = budget - batch_size*self.pilot_cost
            gam_mean_proj, gam_std_proj, log_sig_mean_proj, log_sig_std_proj = self.project_gg_mvn_posterior(batch_size, y_pilot, return_samples=False, project_sig_means=True, wishart=True, winsor=winsor, seed=seed)
            # since we are not projecting log_sig_means, use *current* posterior log sig mean!!
            # actually, we want to equate the mean of the exp(log_sigs), 
            #   as in E[exp(log_sigs)]= exp(mu+(sig**2)/2),
            #   which gives constant E[exp(log_sigs)] via mu+(sig**2)/2 = mu_proj+(sig_proj**2)/2
            log_sig_means_post_proj_new_formula = log_sig_post[0] + (log_sig_post[1]**2 - log_sig_std_proj**2)/2
            # print(log_sig_means_post_proj_new_formula)
            # print(np.exp(log_sig_post[0]+log_sig_post[1]**2/2)**2)
            # print(np.exp(log_sig_means_post_proj_new_formula+log_sig_std_proj[1]**2/2)**2)
            # print(gam_post[0])
            # print(gam_mean_proj)
            # print(np.diag(gam_post[1]))
            # print(np.diag(gam_std_proj))
            # print(log_sig_post[0])
            # print(log_sig_mean_proj)
            gam_samps_k, corr_samps_k, sig_samps_k, cov_samps_k = self.generate_gamma_gaussian_samps_mvn(n_mc, mu_post=gam_mean_proj,
            Sig_post=gam_std_proj,
            log_sig_means_post=log_sig_post[0],#log_sig_mean_proj,#,#log_sig_post[0],
            log_sig_sds_post=log_sig_std_proj,
            seed=seed)  
            
            # choose Sigma prime
            if sigma_prime == 'mean':
                sigma_p_k = cov_samps_k.mean(axis=2)
            elif sigma_prime == 'median':
                sigma_p_k = np.median(cov_samps_k,axis=2)
            elif sigma_prime == 'c_hat':
                sigma_p = c_hat.cpu().detach().numpy()
            elif sigma_prime == 'analytical_mean':
                corr_p_k = corr_utils.GFT_inverse_mapping(gam_mean_proj)[0]
                sds_p_k = np.exp(log_sig_means_post_proj_new_formula)
                sigma_p_k = np.diag(sds_p_k) @ corr_p_k @ np.diag(sds_p_k)
                
            # print( sigma_p)
            # print( sigma_p_k)
            
            if estimator == 'gacv':
                l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss_gacv(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)
            elif estimator == 'acv':
                l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)
            
            # record each step's data
            l_list.append(l)
            losses_list.append(losses)
            l_k_list.append(l_k)
            losses_k_list.append(losses_k)
            budget_list.append(budget)
            print(l,l_k)
            
            # stopping criterion
            if l<l_k:
                print("Termination met at iteration "+str(ii))
                if not estimate:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
                else:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k,seed=seed)
            
            ii+=batch_size
        print("No termination criterion met, terminating pilot sampling at end condition iteration "+str(ii))
        if not estimate:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
        else:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k, seed=seed)    


    def find_n_star_gg_mvn_steps(self, x, 
                                 k=2,
                                 n_steps=5, 
                                 start=None, 
                                 stop=50, 
                                 n_mc=200, 
                                 seed=None, 
                                 y_pilot_precomputed=None, 
                                 sigma_prime='mean', 
                                 estimator='acv', 
                                 winsor=False, 
                                 estimate=False, 
                                 precomputed=False):
        """
        Run gamma gaussian MVN based pilot sampling termination algorithm.
    
        Args:
            x (tensor): Input variables for the model.
            k (int, optional): step size for projected posterior. Default is 2.
            n_steps (int, optional): number of pilot sample steps to project posteriors. Default is 5.
            start (int, optional): Minimum number of iterations. Default is self.n_models + 1.
            stop (int, optional): Maximum number of iterations. Default is 50.
            n_mc (int, optional): Number of MC samples for loss estimation. Default is 200.
            seed (int, optional): Random seed for reproducibility. Default is self.seed.
            sigma_prime (string, optional): Choice of point estimate for \Sigma. Default is posterior mean.
        """
        if start is None:
            start = self.n_models + 1
        if seed is None:
            seed = self.seed
        if self.gamma_mvn_prior_initialized is False:
            print("Please initialize a gamma gaussian MVN prior first!")
            return 0
        
        l_list, l_k_list, losses_list, losses_k_list, budget_list = [], [], [], [], []
        ii = start
        
        models = self.model_utils.return_list_of_models(self.config["checkpoint_dir"], self.config)
    
        while ii<stop:
            pilot_budget = ii * self.pilot_cost
            budget = self.total_budget - pilot_budget
            print("Running projected loss loop for batch size "+str(ii)+" and ACV budget "+str(budget)[:5])

            if precomputed and y_pilot_precomputed is not None:
                y_pilot = torch.Tensor(y_pilot_precomputed[:ii, :].T)
            else:
                _, _, _, _, y_pilot = self.model_utils.model_eval_seed(ii, self.grid, models, self.config, seed=seed)
                y_pilot = y_pilot.detach().T
            c_hat = torch.cov(y_pilot, correction=1)
            # compute and sample from current posterior p(Sigma|y_pilot)
            gam_post = self.gamma_updates_mvn(y_pilot, winsor=winsor, seed=seed)
            log_sig_post = self.log_sigma_updates(y_pilot, winsor=winsor, seed=seed)
            gam_samps, corr_samps, sig_samps, cov_samps = self.generate_gamma_gaussian_samps_mvn(n_mc, mu_post=gam_post[0],
                                            Sig_post=gam_post[1],
                                            log_sig_means_post=log_sig_post[0],
                                            log_sig_sds_post=log_sig_post[1],
                                            seed=seed)
            
            # choose Sigma prime
            if sigma_prime == 'mean':
                sigma_p = cov_samps.mean(axis=2)
            elif sigma_prime == 'median':
                sigma_p = np.median(cov_samps,axis=2)
            elif sigma_prime == 'c_hat':
                sigma_p = c_hat.cpu().detach().numpy()
            elif sigma_prime == 'analytical_mean':
                corr_p = corr_utils.GFT_inverse_mapping(gam_post[0])[0]
                sds_p = np.exp(log_sig_post[0])
                sigma_p = np.diag(sds_p) @ corr_p @ np.diag(sds_p)
            else:
                print("Please select a valid sigma_prime method!")
                return 0
    
            # checking current expected loss
            if estimator == 'gacv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss_gacv(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            elif estimator == 'acv':
                l, losses, var_c_hats, var_opts = self.compute_expected_total_loss(sigma_p, cov_samps, budget, self.total_budget, self.w, return_vec=True)
            
            # projecting next posterior after batch_size additional samples
            l_k_steps = []
            losses_k_steps = []
            for j in range(n_steps):
                k_step = k * (j + 1)
                budget_k = budget - k_step * self.pilot_cost
                gam_mean_proj, gam_std_proj, log_sig_mean_proj, log_sig_std_proj = self.project_gg_mvn_posterior(k_step, y_pilot, return_samples=False, project_sig_means=True, wishart=True, winsor=winsor, seed=seed)
            
                log_sig_means_post_proj_new_formula = log_sig_post[0] + (log_sig_post[1]**2 - log_sig_std_proj**2)/2
            
                gam_samps_k, corr_samps_k, sig_samps_k, cov_samps_k = self.generate_gamma_gaussian_samps_mvn(n_mc, mu_post=gam_mean_proj,
                Sig_post=gam_std_proj,
                log_sig_means_post=log_sig_post[0],
                log_sig_sds_post=log_sig_std_proj,
                seed=seed)  
            
                # choose Sigma prime
                if sigma_prime == 'mean':
                    sigma_p_k = cov_samps_k.mean(axis=2)
                elif sigma_prime == 'median':
                    sigma_p_k = np.median(cov_samps_k,axis=2)
                elif sigma_prime == 'c_hat':
                    sigma_p = c_hat.cpu().detach().numpy()
                elif sigma_prime == 'analytical_mean':
                    corr_p_k = corr_utils.GFT_inverse_mapping(gam_mean_proj)[0]
                    sds_p_k = np.exp(log_sig_means_post_proj_new_formula)
                    sigma_p_k = np.diag(sds_p_k) @ corr_p_k @ np.diag(sds_p_k)
                
            
                if estimator == 'gacv':
                    l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss_gacv(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)
                elif estimator == 'acv':
                    l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss(sigma_p_k, cov_samps_k, budget_k, self.total_budget, self.w, return_vec=True)

                l_k_steps.append(l_k)
                losses_k_steps.append(losses_k)
            
            # record each step's data
            l_list.append(l)
            losses_list.append(losses)
            # l_k_list.append(l_k)
            # losses_k_list.append(losses_k)
            l_k_list.append(l_k_steps)
            losses_k_list.append(losses_k_steps)
            budget_list.append(budget)
            print(l,l_k_steps)
            
            # if l<l_k:

            # compare against multiple step projected losses
            if all(l < l_step for l_step in l_k_steps):
                print("Termination met at iteration "+str(ii))
                if not estimate:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
                else:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k,seed=seed)
            
            ii+=k
        print("No termination criterion met, terminating pilot sampling at end condition iteration "+str(ii))
        if not estimate:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
        else:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k, seed=seed)


    def run_informed_iw_prior_loop(self, x, S_prior, batches, nu_prior=1, batch_size=1, stop=50, n_samps=200, plot=True):
        """
        Run the informed IW prior loop for budget allocation.

        Args:
            x (tensor): Input variables for the model.
            S_prior (numpy array): Prior covariance matrix.
            batches (list): List of batch sizes for sampling.
            nu_prior (int, optional): Prior degrees of freedom. Default is 1.
            batch_size (int, optional): Size of each batch for sampling. Default is 1.
            stop (int, optional): Maximum number of iterations. Default is 50.
            n_samps (int, optional): Number of samples for covariance estimation. Default is 200.
            plot (bool, optional): If True, plots the results.
        """
        if self.iw_initialized == False:
            print("There is no initialized IW prior! Call PilotStudy.initialize_prior()!")
            return
        
        batch_results, batch_budgets = [], []

        for batch in batches:
            print("Running oracle loop for batch size: "+str(batch))
            EI_list, pilot_budget_list = [], []
            initial_budget = batch * self.pilot_cost
            budget = self.total_budget - initial_budget
            y_pilot = self.model_utils.model_eval_seed(x, batch, seed=self.seed).T
            c_hat = torch.cov(y_pilot, correction=0)
            pilot_budget = initial_budget

            with alive_bar(len(range(batch, stop, batch_size))) as bar: 
                for i in range(batch, stop, batch_size):
                    bar()
                    N = int(np.round(pilot_budget / self.pilot_cost))
                    nu_post = nu_prior + N
                    S_post = S_prior + c_hat.cpu().detach().numpy() * N
                    iw_mean = S_post / (nu_post - self.n_models - 1)
                    cov_samps_wis = torch.from_numpy(stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_samps, random_state=self.seed))
                    budget = self.total_budget - pilot_budget
                    EI = self.compute_expected_total_loss(c_hat, torch.swapdims(cov_samps_wis, 0, 2), budget, self.total_budget, self.w, return_vec=False)
    
                    EI_list.append(EI)
                    pilot_budget_list.append(pilot_budget)
                    pilot_budget += batch_size * self.pilot_cost

            batch_results.append(EI_list)
            batch_budgets.append(pilot_budget_list)

        self.batch_results = batch_results
        self.batch_budgets = batch_budgets
        
        if plot:
            self.plot_pilot_results(batches, batch_results, batch_budgets)
            
    def find_n_star_steps(self, x=None, S_prior=None, nu_prior=None, start=None, stop=None, k=2,
                    n_steps=5,
                    n_mc=200, seed=None, y_pilot_precomputed=None, sigma_prime='iw_mean', estimate=False, precomputed=False):
        """
        Run the informed IW prior loop for budget allocation.

        Args:
            x (tensor): Input variables for the model.
            S_prior (numpy array): Prior covariance matrix.
            batches (list): List of batch sizes for sampling.
            nu_prior (int, optional): Prior degrees of freedom. Default is 1.
            k (int, optional): step size for projected posterior. Default is 2.
            n_steps (int, optional): number of pilot sample steps to project posteriors. Default is 5.
            stop (int, optional): Maximum number of iterations. Default is 50.
            n_samps (int, optional): Number of samples for covariance estimation. Default is 200.
            plot (bool, optional): If True, plots the results.
        """
        if self.iw_initialized == False:
            print("There is no initialized IW prior! Call PilotStudy.initialize_prior()!")
            return
        if S_prior is None:
            S_prior = self.S_prior
        if nu_prior is None:
            nu_prior = self.nu_prior
        if start is None:
            start = self.n_models+1
        if stop is None:
            stop = int(self.total_budget/np.sum(self.w))
        if seed is None:
            seed=self.seed
        if x is None:
            x=self.x
            
        l_list, l_k_list, losses_list, losses_k_list, budget_list = [], [], [], [], []
        ii = start

        models = self.model_utils.return_list_of_models(self.config["checkpoint_dir"], self.config)

        while ii<stop:           
            pilot_budget = ii * self.pilot_cost
            budget = self.total_budget - pilot_budget
            print("Running projected loss loop for pilot sample size: "+str(ii)+" and ACV budget "+str(budget)[:5])
            # y_pilot = self.model_utils.model_eval_seed(x, ii, seed=seed).T
            if precomputed and y_pilot_precomputed is not None:
                y_pilot = torch.Tensor(y_pilot_precomputed[:ii, :].T)
            else:
                _, _, _, _, y_pilot = self.model_utils.model_eval_seed(ii, self.grid, models, self.config, seed=seed)
                y_pilot = y_pilot.detach().T
            c_hat = torch.cov(y_pilot, correction=1)
            S = torch.cov(y_pilot, correction=0)
            
            # compute and sample from current posterior p(Sigma|y_pilot)
            nu_post = nu_prior + ii
            S_post = S_prior + S.cpu().detach().numpy() * ii
            S_post = (S_post + S_post.T)/2
            np.random.seed(self.seed)
            cov_samps_wis = torch.from_numpy(stats.invwishart.rvs(df=nu_post, scale=S_post, size=n_mc, random_state=seed))
            
            # choose Sigma'
            if sigma_prime == 'iw_mean':
                sigma_p = S_post / (nu_post - self.n_models - 1)
            elif sigma_prime == 'iw_mode':
                sigma_p = S_post / (nu_post + self.n_models + 1)
            elif sigma_prime == 'S':
                sigma_p = S.cpu().detach().numpy()
            elif sigma_prime == 'c_hat':
                sigma_p = c_hat.cpu().detach().numpy()
            else:
                print("Please select a valid sigma_prime method!")
                return 0

            # checking current expected loss
            l, losses, var_c_hats, var_opts = self.compute_expected_total_loss(sigma_p, torch.swapdims(cov_samps_wis, 0, 2), budget, self.total_budget, self.w, return_vec=True)
            
            # projecting next posterior after k * n_steps additional samples

            l_k_steps = []
            losses_k_steps = []

            for j in range(n_steps):
                k_step = k * (j + 1)
                nu_post_proj = nu_post + k_step
                #prev_mean_cov = S_post/(nu_post-self.n_models-1)
                #S_post_proj = S_post + prev_mean_cov * batch_size
                S_post_proj = S_prior + S.cpu().detach().numpy() * (ii + k_step)
                budget_k = budget - k_step *self.pilot_cost
                np.random.seed(self.seed)
                cov_samps_wis_k = torch.from_numpy(stats.invwishart.rvs(df=nu_post_proj, scale=S_post_proj, size=n_mc, random_state=seed))
                # choose Sigma'
                if sigma_prime == 'iw_mean':
                    sigma_p_k = S_post_proj / (nu_post_proj - self.n_models - 1)
                elif sigma_prime == 'iw_mode':
                    sigma_p_k = S_post_proj / (nu_post_proj + self.n_models + 1)
                elif sigma_prime == 'S':
                    sigma_p_k = S.cpu().detach().numpy()
                elif sigma_prime == 'c_hat':
                    sigma_p_k = c_hat.cpu().detach().numpy()
                
                #print(sigma_p, sigma_p_k)

                l_k, losses_k, var_c_hats_k, var_opts_k = self.compute_expected_total_loss(sigma_p_k, torch.swapdims(cov_samps_wis_k, 0, 2), budget_k, self.total_budget, self.w, return_vec=True)
            
                l_k_steps.append(l_k)
                losses_k_steps.append(losses_k)
            # record each step's data
            l_list.append(l)
            losses_list.append(losses)
            # l_k_list.append(l_k)
            # losses_k_list.append(losses_k)

            l_k_list.append(l_k_steps)
            losses_k_list.append(losses_k_steps)

            budget_list.append(budget)
            print(l, l_k_steps)
            # print(f"Current losses: {l:.5e}; projected losses: {l_k:.5e}")
            #print("Current losses: "+str(l)[:7]+"; projected losses: "+str(l_k)[:7])
            
            # stopping criterion
            # if l<l_k:
            if all(l < l_step for l_step in l_k_steps):
                print("Termination met at iteration "+str(ii))
                if not estimate:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
                else:
                    return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k,seed=seed)
            
            ii+=k
        print("No termination criterion met, terminating pilot sampling at end condition iteration "+str(ii))
        if not estimate:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats
        else:
            return np.array(l_list), np.array(losses_list), np.array(l_k_list), np.array(losses_k_list), np.array(budget_list), sigma_p, budget_k, y_pilot, var_c_hats, self.acv(sigma_p, budget_k, seed=seed)
            
    def acv(self, cov, b, x=None, seed=None):
        
        if seed is None:
            seed = self.seed
        if x is None:
            x = self.x
        
        w = self.w.detach().numpy().flatten()
        if torch.is_tensor(cov):
            cov = cov.detach().numpy() 
        
        variance_results = dict()
        sample_allocation_results = dict()
        mxmc_optimizer = Optimizer(w, cov)
        algorithms = list(Optimizer.get_algorithm_names())
        
        for algorithm in ["wrdiff"]: # fix this later!
            if algorithm == ('mfmc', 'mlmc'):
                variance_results[algorithm] = 999999.
            else:
                opt_result = mxmc_optimizer.optimize(algorithm, b)
                variance_results[algorithm] = opt_result.variance
                sample_allocation_results[algorithm] = opt_result.allocation

        best_method_pilot = min(variance_results, key=variance_results.get)
        sample_allocation = sample_allocation_results[best_method_pilot]
        estimator = Estimator(sample_allocation, cov)
        
        num_total_samples = sample_allocation.num_total_samples
        all_samples = self.model_utils.z_sample_seed(x, num_total_samples, seed=seed).T
        model_input_samples = sample_allocation.allocate_samples_to_models(all_samples)
        
        # models = self.model_utils.return_list_of_models()
        models = self.model_utils.return_list_of_models(self.config["checkpoint_dir"], self.config)
        # model_outputs = list()
        model_outputs = self.model_utils.model_eval_from_samples(model_input_samples, self.grid, models, self.config)

        # for input_sample, model in zip(model_input_samples, models):
        #     model_outputs.append(model(input_sample))
        
        # MXMC's Estimator can be used to analyse the model outputs
        # to produce an estimate of the quantity of interest.
        estimator = Estimator(sample_allocation, cov)
        return estimator.get_estimate(model_outputs)


    def compute_oracle_loss(self, c_hat_samps, oracle_cov, b_acv, b_full, return_vec=False):
        """
        Compute the oracle loss based on posterior covariance samples.

        Args:
            c_hat_samps (torch.Tensor): n_models x n_models x n_samps array of posterior c_hat samples.
            oracle_cov (np.ndarray): n_models x n_models array representing the oracle covariance.
            b_acv (float): Budget allocated for ACV estimation.
            b_full (float): Total available budget.
            return_vec (bool, optional): If True, returns full vector of cost losses.

        Returns:
            float: Mean of oracle losses over covariance samples.
            Optional (if return_vec is True): Tuple containing the mean oracle loss, individual losses, 
                                              variance estimates for c_hats, 
                                              and variance estimates for the optimal loss.
        """
        # Local numpy array conversion
        w = self.w.detach().numpy().flatten() 
        c_hat_samps = c_hat_samps.detach().numpy() if torch.is_tensor(c_hat_samps) else c_hat_samps
        oracle_cov = oracle_cov.detach().numpy() if torch.is_tensor(oracle_cov) else oracle_cov
        b_acv = b_acv.detach().numpy() if torch.is_tensor(b_acv) else b_acv
        b_full = b_full.detach().numpy() if torch.is_tensor(b_full) else b_full
        
        n_samps = c_hat_samps.shape[2]
        oracle_losses = np.zeros((n_samps,))
        
        # construct the oracle best estimator
        variance_results_oracle = dict()
        sample_allocation_results_oracle = dict()
        mxmc_optimizer = Optimizer(w, oracle_cov)
        algorithms = list(Optimizer.get_algorithm_names())
        for algorithm in ["wrdiff"]:#algorithms:
            if algorithm=='mfmc' or algorithm=='mlmc':
                variance_results_oracle[algorithm] = 999999.
            else:
                opt_result = mxmc_optimizer.optimize(algorithm, b_full)
                variance_results_oracle[algorithm] = opt_result.variance
                sample_allocation_results_oracle[algorithm] = opt_result.allocation
                #print("{} method avg. variance: {}".format(algorithm, var_mean_results[algorithm]))
        best_method_oracle = min(variance_results_oracle, key=variance_results_oracle.get)
        sample_allocation_oracle = sample_allocation_results_oracle[best_method_oracle]
        #print("best method: "+best_method_pilot)
        estimator_oracle = Estimator(sample_allocation_oracle, oracle_cov)
        
        # construct estimator for c_hat, b_acv
        estimator_c_hat_list = []
        sample_allocation_c_hat_list = []
        for samp in range(n_samps):
            c_hat = c_hat_samps[:,:,samp]
            variance_results = dict()
            sample_allocation_results = dict()
            mxmc_optimizer = Optimizer(w, c_hat)
            algorithms = list(Optimizer.get_algorithm_names())
            for algorithm in ["wrdiff"]:#algorithms:
                if algorithm=='mfmc' or algorithm=='mlmc':
                    variance_results[algorithm] = 999999.
                else:
                    opt_result = mxmc_optimizer.optimize(algorithm, b_acv)
                    variance_results[algorithm] = opt_result.variance
                    sample_allocation_results[algorithm] = opt_result.allocation
                    #print("{} method avg. variance: {}".format(algorithm, var_mean_results[algorithm]))
            best_method_pilot = min(variance_results, key=variance_results.get)
            if best_method_pilot=='mfmc':
                print(samp)
                print(variance_results)
                print(c_hat)
            sample_allocation_pilot = sample_allocation_results[best_method_pilot]
            #print("best method: "+best_method_pilot)
            estimator_c_hat_list.append(Estimator(sample_allocation_pilot, c_hat))
            sample_allocation_c_hat_list.append(sample_allocation_pilot)

        for samp in range(n_samps):
            # compute var[Q_oracle]
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_oracle, oracle_cov)
            alpha = estimator_oracle._calculate_alpha()
            var_sf = oracle_cov[0,0]/sample_allocation_oracle.get_number_of_samples_per_model()[0]
            var_oracle =  var_sf + alpha@cov_delta_delta@alpha + 2*alpha@cov_q_delta
            
            #  compute true var[Q_acv,c_hat_samp] at oracle covariance
            cov_delta_delta, cov_q_delta = self.calculate_cov_delta_terms(estimator_c_hat_list[samp], oracle_cov)
            alpha = estimator_c_hat_list[samp]._calculate_alpha()
            var_sf = oracle_cov[0,0]/sample_allocation_c_hat_list[samp].get_number_of_samples_per_model()[0]
            var_c_hat =  var_sf + alpha@cov_delta_delta@alpha + 2*alpha@cov_q_delta

            oracle_losses[samp] = var_c_hat - var_oracle

        return (oracle_losses.mean(), oracle_losses) if return_vec else oracle_losses.mean()
    
    def run_oracle_loop(self, method='iw', initial_batch=4, batch_size=1, n_samps=300, stop=50):
        """
        Run the oracle loop to analyze step-by-step losses.

        Args:
            method (string): ACV design based on IW posterior mean ("iw") or sample cov ("sample_cov") or analytical nonlinear shrinkage of sample cov ("nls-ana-sample-cov") or linear shrinkage ("ls-ana-sample-cov")
            initial_batch (int): Initial batch size for pilot sampling.
            batch_size (int): Batch size to iterate over (resolution of N_pilot plot/data)
            n_samps (int): Number of posterior samples for covariance.
            stop (int): Maximum number of iterations.
        """
        cov_oracle = self.model_utils.sig_func(self.x)
        nu_prior, S_prior = self.nu_prior, self.S_prior
        initial_budget = initial_batch * self.pilot_cost
        
        oracle_loss_list, oracle_lb_list, oracle_ub_list, pilot_budget_list = [], [], [], []
        pilot_budget = initial_budget
        
        c_hats_shrinkage_all = []
        c_hats_all = []

        with alive_bar(len(range(initial_batch, stop, batch_size))) as bar: 
            for i in range(initial_batch, stop, batch_size):
                bar()
                N = int(np.round(pilot_budget / self.pilot_cost))
                c_hats, iw_means = torch.zeros((self.n_models, self.n_models, n_samps)), np.zeros((self.n_models, self.n_models, n_samps))
                c_hats_ana_nls = torch.zeros((self.n_models, self.n_models, n_samps))
                c_hats_ana_ls = torch.zeros((self.n_models, self.n_models, n_samps))
                
                for samp in range(n_samps):
                    y_pilot = self.model_utils.model_eval_seed(self.x, N, seed=samp).T
                    c_hats[:, :, samp] = torch.cov(y_pilot, correction=0)
                    
                    if method == 'nls-ana-sample-cov':
                        # c_hats_ana_nls[:, :, samp] = corr_utils.cov_nl_shrinkage_ana(c_hats[:, :, samp], n=N)
                        # use scikit-rmt method
                        c_hats_ana_nls[:, :, samp] = corr_utils.analytical_shrinkage_estimator(y_pilot.T.detach().numpy())
                    elif method == 'ls-ana-sample-cov':
                        c_hats_ana_ls[:, :, samp] = corr_utils.linear_shrinkage_estimator(y_pilot.T.detach().numpy())

                    nu_post = nu_prior + N
                    S_post = S_prior + c_hats[:, :, samp].cpu().detach().numpy() * N
                    iw_means[:, :, samp] = S_post / (nu_post - self.n_models - 1)
                
                c_hats_shrinkage_all.append((c_hats_ana_nls, c_hats_ana_ls))
                c_hats_all.append(c_hats)


                budget = self.total_budget - pilot_budget
                
                if method == 'iw':
                    oracle_loss, oracle_losses = self.compute_oracle_loss(iw_means, cov_oracle, budget, self.total_budget, return_vec=True)
                elif method == 'sample_cov':
                    oracle_loss, oracle_losses = self.compute_oracle_loss(c_hats, cov_oracle, budget, self.total_budget, return_vec=True)
                elif method == 'nls-ana-sample-cov':
                    oracle_loss, oracle_losses = self.compute_oracle_loss(c_hats_ana_nls, cov_oracle, budget, self.total_budget, return_vec=True)
                    self.c_hats_ana_nls = c_hats_ana_nls
                elif method == 'ls-ana-sample-cov':
                    oracle_loss, oracle_losses = self.compute_oracle_loss(c_hats_ana_ls, cov_oracle, budget, self.total_budget, return_vec=True)
                    self.c_hats_ana_ls = c_hats_ana_ls
                else:
                    raise ValueError("Invalid method, valid methods are: iw, sample_cov, ls-ana-sample-cov, or nls-ana-sample-cov.")
                
                oracle_loss_list.append(oracle_loss)
                pilot_budget_list.append(pilot_budget)
                oracle_lb_list.append(np.quantile(oracle_losses, 0.1))
                oracle_ub_list.append(np.quantile(oracle_losses, 0.9))
                pilot_budget += batch_size * self.pilot_cost
    
        self.plot_oracle_results_nology(pilot_budget_list, oracle_loss_list, oracle_lb_list, oracle_ub_list, method=method)
        self.oracle_budget_list = pilot_budget_list
        self.oracle_loss_list  = oracle_loss_list
        self.oracle_lb_list = oracle_lb_list
        self.oracle_ub_list = oracle_ub_list
        # self.plot_oracle_results(pilot_budget_list, oracle_loss_list, oracle_lb_list, oracle_ub_list)
        
        return pilot_budget_list, oracle_loss_list, oracle_lb_list, oracle_ub_list

    def plot_oracle_results(self, pilot_budget_list, oracle_loss_list, oracle_lb_list, oracle_ub_list, method='iw'):
        """
        Plot the results of the oracle analysis.

        Args:
            pilot_budget_list (list): List of pilot budget values used in sampling.
            oracle_loss_list (list): List of mean oracle losses.
            oracle_lb_list (list): List of lower bound quantiles for losses.
            oracle_ub_list (list): List of upper bound quantiles for losses.
        """
        plt.rcParams['figure.dpi'] = 600
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(100*np.array(pilot_budget_list)/self.total_budget, oracle_loss_list, label='Mean Loss')
        ax.fill_between(100*np.array(pilot_budget_list)/self.total_budget, oracle_ub_list, oracle_lb_list, label=r'$0.9$ Quantile', alpha=0.2)
        ax.set_yscale('log')
        ax.set_xlabel("Percent of Total Budget Used on Pilot Sampling")
        ax.set_ylabel("Expected Loss (variance)")
        if method=='iw':
            ax.set_title("Empirical Losses, IW Posterior Mean Method")
        elif method=='siw':
            ax.set_title("Empirical Losses, SIW Posterior Mean Method")
        elif method=='sample_cov':
            ax.set_title("Empirical Losses, Sample Covariance Method")
        elif method=='nls-ana-sample-cov':
            ax.set_title("Empirical Losses, Non-Linear Shrinkage Method")
        elif method=='ls-ana-sample-cov':
            ax.set_title("Empirical Losses, Linear Shrinkage Method")
        ax.legend()
        plt.show()
        plt.savefig("oracle_results_plot.png")  # Save the plot as an image file
        plt.close(fig)  # Close the figure to avoid display

        self.oracle_fig = fig  # Save figure for later reference
        self.oracle_ax = ax
        
    def plot_oracle_results_nology(self, pilot_budget_list, oracle_loss_list, oracle_lb_list, oracle_ub_list, method='iw'):
        """
        Plot the results of the oracle analysis.

        Args:
            pilot_budget_list (list): List of pilot budget values used in sampling.
            oracle_loss_list (list): List of mean oracle losses.
            oracle_lb_list (list): List of lower bound quantiles for losses.
            oracle_ub_list (list): List of upper bound quantiles for losses.
        """
        plt.rcParams['figure.dpi'] = 600
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(100*np.array(pilot_budget_list)/self.total_budget, oracle_loss_list, label='Mean Loss')
        ax.fill_between(100*np.array(pilot_budget_list)/self.total_budget, oracle_ub_list, oracle_lb_list, label=r'$0.9$ Quantile', alpha=0.2)
        #ax.set_yscale('log')
        ax.set_xlabel("Percent of Total Budget Used on Pilot Sampling")
        ax.set_ylabel("Expected Loss (variance)")
        if method=='iw':
            ax.set_title("Empirical Losses, IW Posterior Mean Method")
        elif method=='siw':
            ax.set_title("Empirical Losses, SIW Posterior Mean Method")
        elif method=='sample_cov':
            ax.set_title("Empirical Losses, Sample Covariance Method")
        elif method=='nls-ana-sample-cov':
            ax.set_title("Empirical Losses, Non-Linear Shrinkage Method")
        elif method=='ls-ana-sample-cov':
            ax.set_title("Empirical Losses, Linear Shrinkage Method")
        ax.legend()
        plt.show()
        plt.savefig("oracle_results_plot_nology.png")  # Save the plot as an image file
        plt.close(fig)  # Close the figure to avoid display

        self.oracle_fig_nology = fig  # Save figure for later reference
        self.oracle_ax_nology = ax

    def plot_pilot_results(self, batches, batch_results, batch_budgets,method='iw'):
        """
        Plot the results of the informed IW prior loop.

        Args:
            batches (list): List of batch sizes.
            batch_results (list): List of results for each batch size.
            batch_budgets (list): List of budgets for each batch size.
        """
        plt.rcParams['figure.dpi'] = 600
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('Blues')
        j = 0
        for budget, result in zip(batch_budgets, batch_results):
            ax.semilogy(100*np.array(budget)/self.total_budget, np.array(result), color=cmap((j+10) / (len(batches)+10)))
            j += 1
        ax.set_xlabel("Percent of Total Budget Used on Pilot Sampling")
        ax.set_ylabel("Expected Loss (variance)")
        ax.legend([f"Batch {batch}" for batch in batches])
        if method=='iw':
            ax.set_title("Expected Loss Projections, IW-Mean Estimator")
        elif method=='sample_cov':
            ax.set_title("Expected Loss Projections, Sample Covariance Estimator")
        plt.show()
        plt.savefig("pilot_results_plot.png")  # Save the plot as an image file
        plt.close(fig)  # Close the figure to avoid display

        self.pilot_fig = fig  # Save figure for later reference
        self.pilot_ax = ax
        
    def plot_pilot_results_nology(self, batches, batch_results, batch_budgets):
        """
        Plot the results of the informed IW prior loop.

        Args:
            batches (list): List of batch sizes.
            batch_results (list): List of results for each batch size.
            batch_budgets (list): List of budgets for each batch size.
        """
        plt.rcParams['figure.dpi'] = 600
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        j = 0
        for budget, result in zip(batch_budgets, batch_results):
            ax.plot(100*np.array(budget)/self.total_budget, np.array(result), color=cmap(j / (len(batches))))
            j += 1
        ax.set_xlabel("Percent of Total Budget Used on Pilot Sampling")
        ax.set_ylabel("Expected Loss (variance)")
        ax.legend([f"Batch {batch}" for batch in batches])
        ax.set_title("Expected Loss Projections using IW Posteriors")
        plt.show()
        plt.savefig("pilot_results_plot_nology.png")  # Save the plot as an image file
        plt.close(fig)  # Close the figure to avoid display

        self.pilot_fig_nology = fig  # Save figure for later reference
        self.pilot_ax_nology = ax
        
    def plot_pilot_results_SIW(self, batches, batch_results, batch_budgets):
        """
        Plot the results of the informed IW prior loop.

        Args:
            batches (list): List of batch sizes.
            batch_results (list): List of results for each batch size.
            batch_budgets (list): List of budgets for each batch size.
        """
        plt.rcParams['figure.dpi'] = 600
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        j = 0
        for budget, result in zip(batch_budgets, batch_results):
            ax.semilogy(100*np.array(budget)/self.total_budget, np.array(result), color=cmap(j / (len(batches))))
            j += 1
        ax.set_xlabel("Percent of Total Budget Used on Pilot Sampling")
        ax.set_ylabel("Expected Loss (variance)")
        ax.legend([f"Batch {batch}" for batch in batches])
        ax.set_title("Expected Loss Projections using SIW Posteriors")
        plt.show()
        plt.savefig("pilot_results_plot_SIW.png")  # Save the plot as an image file
        plt.close(fig)  # Close the figure to avoid display

        self.pilot_fig = fig  # Save figure for later reference
        self.pilot_ax = ax
