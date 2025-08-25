# %%
import os
import sys
import numpy as np
import torch
sys.path.insert(0, "../src")
from PilotStudy_PDE import PilotStudy
import corr_utils
from unet_fno import UNet2d, FNO2d
import model_utils
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, invwishart
plt.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')
import scipy.stats as stats
import yaml
from multiprocessing import Pool, cpu_count

n_cpus = cpu_count()
print("Number of CPUs available:", n_cpus)

yaml_file = os.path.join("../data/", "config_2D_train.yaml")
with open(yaml_file, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

num_channels = yaml_data["num_channels"]
modes = yaml_data["modes"]
width = yaml_data["width"]
initial_step = yaml_data["initial_step"]
in_channels = yaml_data["in_channels"]
out_channels = yaml_data["out_channels"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_lf1 = UNet2d(in_channels=in_channels,
               out_channels=out_channels).to(device)
model_lf2 = FNO2d(num_channels=num_channels,
              modes1=modes, 
              modes2=modes, 
              width=width, 
              initial_step=initial_step).to(device)


grid_single = torch.Tensor(np.load("../data/grid_xy_darcy.npy")[0, :, :, :])
ckpt_lf1 = os.path.join("../data", "checkpoints", "unet_darcy_beta_1.pt")
ckpt_lf2 = os.path.join("../data", "checkpoints", "fno_darcy_beta_1.pt")

# %%
x = torch.tensor([.5])
n_models = 3
w = torch.tensor([[1.],[0.006],[0.004]]) # model costs
pilot_cost = w.sum().item()
total_budget = 200*pilot_cost
seed = 35
pilot_study = PilotStudy(model_utils=model_utils, 
                         x=x, 
                         total_budget=total_budget,  
                         n_models=n_models, 
                         w=w,
                         grid=grid_single,
                         yaml_data=yaml_data,
                         seed=seed)

models = pilot_study.model_utils.return_list_of_models(pilot_study.config["checkpoint_dir"], pilot_study.config)

# %%
pilot_case = "informative"
start_pilot = 4
stop_pilot = 100

k = 2
n_steps = 6
# %%
if pilot_case == "informative":
    mean_corr_prior = np.array([[1., 0.8, 0.7],
                    [0.8, 1.   ,  0.7],
                    [0.7 , 0.7 , 1.]])
    gamma_means = corr_utils.GFT_forward_mapping(mean_corr_prior)
    gamma_cov =  np.diag(np.ones(gamma_means.shape))
    pilot_study.initialize_gamma_mvn_prior(gamma_means, gamma_cov)
    log_sigma_means = np.log(np.array([.05, .05, .05]))
    log_sigma_sigs = np.ones((n_models,))
    pilot_study.initialize_log_sigma_prior(log_sigma_means, log_sigma_sigs)           

elif pilot_case == "non_informative":
    mean_corr_prior = np.array([[1., 0.5, 0.5],
                [0.5, 1.   ,  0.5],
                [0.5 , 0.5 , 1.]])
    gamma_means = corr_utils.GFT_forward_mapping(mean_corr_prior)
    gamma_cov = np.diag(np.ones(gamma_means.shape))
    pilot_study.initialize_gamma_mvn_prior(gamma_means, gamma_cov)
    log_sigma_means = np.log(np.array([.1, .1, .1]))
    log_sigma_sigs = np.ones((n_models,))
    pilot_study.initialize_log_sigma_prior(log_sigma_means, log_sigma_sigs)


# %%
from matplotlib.lines import Line2D
cmap = plt.cm.viridis_r#plt.cm.Blues
colors = cmap(np.linspace(0.3, .9, 13)) # adjust 0.3 to 0.9 for light to dark

y_pilot_pre_proj = np.load("../data/y_pilot_all_precomputed_6026_6030_rel_larger_bounding_box.npy")
y_pilot_pre_proj_perm = (y_pilot_pre_proj.T).T
oracle_corr = np.load("../data/oracle_corr_sf_data.npy")
oracle_stds = np.load("../data/oracle_stds_all_models.npy")
oracle_cov = np.diag(oracle_stds) @ oracle_corr @ np.diag(oracle_stds)

projected_vars = []
actual_vars = []
n_mc=800
seed_hist = 6026
for n_pilot_orig in [6, 8, 12, 20]:
    print(n_pilot_orig)
    y_pilot = y_pilot_pre_proj_perm[:, :n_pilot_orig]
    pilot_gamma = corr_utils.GFT_forward_mapping(np.cov(y_pilot))
    gam_post = pilot_study.gamma_updates_mvn(y_pilot,winsor=True,seed=seed_hist)
    log_sig_post = pilot_study.log_sigma_updates(y_pilot,winsor=True,seed=seed_hist)
    b_acv = total_budget - n_pilot_orig*pilot_cost
    if n_pilot_orig == 6:
        log_sig_val = log_sigma_means
    else:
        log_sig_val = log_sig_post[0]
    gam_samps, corr_samps, sig_samps, cov_samps = pilot_study.generate_gamma_gaussian_samps_mvn(n_mc, mu_post=gam_post[0],
                            Sig_post=gam_post[1],
                            log_sig_means_post=log_sig_val,
                            log_sig_sds_post=log_sig_post[1],
                            seed=seed_hist)
    
    c_hat = np.mean(cov_samps,axis=2)
    l_bar, losses, var_c_hats, var_opts = pilot_study.compute_expected_total_loss(c_hat, cov_samps, b_acv, total_budget, w, return_vec = True)
    actual_var, test = pilot_study.compute_oracle_acv_variance(c_hat, b_acv, oracle_cov, w, estimator='wrdiff', return_projection=True) 
    projected_vars.append(var_c_hats)
    actual_vars.append(actual_var)

# %%
count = 0
hist_max = [375, 325, 275, 225]
savefig = True
for n_pilot_orig in [6, 8, 12, 20]:
    print(count)
    i = np.where(np.arange(6, 50, step=2)==n_pilot_orig)[0].item()
    plt.figure()
    hist = plt.hist(projected_vars[count],label='Projections',color=colors[i],alpha=0.5)
    plt.vlines(actual_vars[count],
            0,
            hist_max[count],
            color='black',
            label='Actual')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Variance', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.title(rf'$N_{{\text{{pilot}}}}={n_pilot_orig}$', fontsize=24)
    plt.xlim([0.,0.15e-4])
    plt.xticks(fontsize=18)
    plt.ylim([0, hist_max[count]])
    plt.yticks(fontsize=18)
    if savefig:
        plt.savefig("fig_9_var_hist_ggmvn_nmc800_n_pilot_{:02d}.png".format(n_pilot_orig), bbox_inches="tight", dpi=300)
        print("Saved fig_9_var_hist_ggmvn_nmc800_n_pilot_{:02d}.png".format(n_pilot_orig))
        plt.close()
    count+=1

# %%