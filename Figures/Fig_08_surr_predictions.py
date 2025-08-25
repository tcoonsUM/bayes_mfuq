import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, "../src")
from unet_fno import UNet2d, FNO2d
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

plt.rc('axes.spines', top=True, right=True)
plt.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')
plt.rcParams.update({'font.family': 'serif'})

# %%
example_path = "../data"
checkpoint_path = "../data/checkpoints"
yaml_file = "../data/config_2D_train.yaml"
with open(yaml_file, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

init_key_train = yaml_data["init_key"]
init_key_test = yaml_data["init_key_test"]
in_channels = yaml_data["in_channels"]
out_channels = yaml_data["out_channels"]
num_channels = yaml_data["num_channels"]
modes = yaml_data["modes"]
width = yaml_data["width"]
initial_step = yaml_data["initial_step"]
# use_point_qoi = False

# %%
holdout_set = np.load("../data/darcy_flow_surrogate_test_set_init_key_start_1994.npz")

x_test_raw = torch.Tensor(holdout_set["nu_all"])
y_test_raw = torch.Tensor(holdout_set["uu_all"])
device = torch.device("cpu")
N_TEST = x_test_raw.shape[0]

grid_single = torch.Tensor(np.load("../data/grid_xy_darcy.npy")[0, :, :, :])
grid_test = grid_single.unsqueeze(0).repeat(N_TEST, 1, 1, 1)

learning_rate = 1e-3
scheduler_step = 5
scheduler_gamma = 0.5
batch_size_test = N_TEST
num_workers = 1

# %%
surr_preds = []
for model_name in ["unet", "fno"]:
    if model_name == "fno":
        model = FNO2d(num_channels=num_channels,
              modes1=modes, 
              modes2=modes, 
              width=width, 
              initial_step=initial_step).to(device)
    elif model_name == "unet":
        model = UNet2d(in_channels=in_channels,
               out_channels=out_channels).to(device)

        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    os.makedirs(os.path.join(example_path, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(example_path, "checkpoints", "{}_darcy_beta_1.pt".format(model_name))

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        raise FileNotFoundError(f"The path '{checkpoint_path}' does not exist, set train model to True")
    if model_name == "fno":
        x_test_sample  = torch.cat((x_test_raw.unsqueeze(-1), grid_test), dim=-1)
    elif model_name == "unet":
        x_test_sample = x_test_raw.unsqueeze(1)
        
    
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test_sample)
    
    surr_preds.append(y_pred.squeeze().cpu().unsqueeze(0))

# %%
ensemble_preds = [y_test_raw, surr_preds[0].squeeze(), surr_preds[1].squeeze()]

# %%
def plot_snapshots_with_bounding_box(imgs_hf,
                                     imgs_lf1,
                                     imgs_lf2,
                                     selected_idx,
                                     yaml_data,
                                     x_min = 0.6,
                                     x_max = 0.8,
                                     y_min = 0.75,
                                     y_max = 0.95,
                                     set_title=True,
                                     savefig=False):

    """
    Supply tensors of shape NSims x Nx x Ny, where Nx and Ny are the grid dimensions. The tensors should be in the same order as the ones returned by the run_hifi_darcy function. The x and y bounds specify the overlaid bounding box.
    """

    img_hf = imgs_hf[selected_idx]
    img_lf1 = imgs_lf1[selected_idx]
    img_lf2 = imgs_lf2[selected_idx]

    pred_min = min(img_hf.min(), img_lf1.min(), img_lf2.min())
    pred_max = max(img_hf.max(), img_lf1.max(), img_lf2.max())


    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    cax = plt.subplot(gs[3])

    im0 = ax0.imshow(img_hf,
                 cmap="viridis",
                 vmin=pred_min,
                 vmax=pred_max,
                 extent=(0, 1, 0, 1)
                )
    if set_title:
        ax0.set_title(r"PDE Solution ($f_0$)", fontsize=18)

    im1 = ax1.imshow(img_lf1,
                 cmap="viridis",
                 vmin=pred_min,
                 vmax=pred_max,
                 extent=(0, 1, 0, 1)
                )
    if set_title:
        ax1.set_title(r"U-net ($f_1$)", fontsize=18)

    im2 = ax2.imshow(img_lf2,
                 cmap="viridis",
                 vmin=pred_min,
                 vmax=pred_max,
                 extent=(0, 1, 0, 1)
                )
    if set_title:
        ax2.set_title(r"FNO ($f_2$)", fontsize=18)

    for ax in [ax0, ax1, ax2]:
        rect = patches.Rectangle((x_min, y_min), 
                        (x_max - x_min), 
                        (y_max - y_min), 
                        linewidth=2.5, 
                        edgecolor='red', 
                        facecolor='none')
        ax.add_patch(rect)

    for ax in [ax0, ax1, ax2]:
        ax.set(xticks=[], yticks=[])


    cbar = fig.colorbar(im2, cax=cax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=22)

    if savefig:
        plt.savefig("snapshots_with_box_final_simID_{:03d}.png".format(selected_idx), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        # return fig
        plt.show()
    
# %%
plot_snapshots_with_bounding_box(ensemble_preds[0],
                                ensemble_preds[1],
                                ensemble_preds[2],
                                5,
                                yaml_data,
                                x_min = 0.6,
                                x_max = 0.8,
                                y_min = 0.75,
                                y_max = 0.95,
                                savefig=False)

# %%
plot_snapshots_with_bounding_box(ensemble_preds[0],
                                ensemble_preds[1],
                                ensemble_preds[2],
                                8,
                                yaml_data,
                                x_min = 0.6,
                                x_max = 0.8,
                                y_min = 0.75,
                                y_max = 0.95,
                                savefig=False)

# %%