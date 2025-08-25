import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rc('axes.spines', top=True, right=True)
plt.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')
plt.rcParams.update({'font.family': 'serif'})

def show_image_panels(imgs, img_min, img_max, disc=False, savefig=False, varstring="u", size=None):
    """
    Given a tensor of shape NSims x Nx x Ny, convert it into a list using something on the lines of `imgs_uu = [img for img in torch.Tensor(uu_all)]` and pass it along with the colorbar limits (img_min, img_max). For the permeability inputs, we also pass the `disc` argument to create a custom discrete colormap with appropriate ticklabels. If size is (None), we get a 15x8 grid with 120 snapshots, else we can pass an appropriate tuple depending on our snapshots.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=8, nrows=15, squeeze=False, figsize=(9, 16))
    
    ref_ax = axs[0, 0]
    im = None
    
    if disc:
        boundaries = [img_min - 0.5, img_min + 0.5, img_max + 0.5]
        norm = BoundaryNorm(boundaries, ncolors=2)
        base_cmap = plt.get_cmap('viridis')
        colors = [base_cmap(0.0), base_cmap(1.0)]  # low and high ends
        discrete_cmap = ListedColormap(colors)
    
    for i, (img, ax) in enumerate(zip(imgs, axs.ravel())):
        img = img.detach()
        img_np = img.cpu().numpy()
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if disc:
            im = ax.imshow(img_np, cmap=discrete_cmap, 
                           norm=norm)
            
        else:
            im = ax.imshow(img_np, cmap="viridis", vmin=img_min, vmax=img_max)

    for ax in axs.ravel()[len(imgs):]:
        ax.axis("off")


    if disc:
        cbar = fig.colorbar(im, 
                            ax=axs, 
                            orientation='vertical', 
                            fraction=0.04, 
                            pad=0.02, 
                            ticks=[img_min, img_max])
        cbar.ax.tick_params(labelsize=22)
    else:
        cbar = fig.colorbar(im, ax=axs, 
                            orientation='vertical', 
                            fraction=0.04, 
                            pad=0.02)
        cbar.ax.tick_params(labelsize=22)

    if savefig:
        plt.savefig("panels_{}.png".format(varstring), dpi=200)
        plt.close()
    else:
        # return fig
        plt.show()
    

# %%
training_data = np.load("../data/darcy_flow_surrogate_training_set_init_key_start_2026.npz")

ic_train = training_data["nu_all"]
u_train = training_data["uu_all"]

imgs_ic = [img for img in torch.Tensor(ic_train)]

ic_min = ic_train.min()
ic_max = ic_train.max()


imgs_u = [img for img in torch.Tensor(u_train)]

u_min = u_train.min()
u_max = u_train.max()

# %%

show_image_panels(imgs_ic, ic_min, ic_max, disc=True, savefig=False, varstring="ic")

show_image_panels(imgs_u, u_min, u_max, disc=False, savefig=False, varstring="u")

# %%

