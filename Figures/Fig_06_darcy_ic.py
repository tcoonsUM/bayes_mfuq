import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rc('axes.spines', top=True, right=True)
plt.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')
plt.rcParams.update({'font.family': 'serif'})

def show_raw_and_final_ic(raw_ic, final_ic, savefig=False, base_dir="/home/ajivani/Covuq/Paper01"):
    """
    raw_ic represents the permeability field features prior to thresholding.
    final_ic represents the binary permeability field after thresholding.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    disc_img_min = 0.1
    disc_img_max = 1.0
    boundaries = [disc_img_min - 0.5, disc_img_min + 0.5, disc_img_max + 0.5]
    norm_disc = BoundaryNorm(boundaries, ncolors=2)
    base_cmap = plt.get_cmap('viridis')
    colors = [base_cmap(0.0), base_cmap(1.0)]
    discrete_cmap = ListedColormap(colors)

    im0 = ax[0].imshow(raw_ic,
                       vmin=raw_ic.min(),
                       vmax=raw_ic.max()
                    )
    im1 = ax[1].imshow(final_ic, 
                        cmap=discrete_cmap, 
                        norm=norm_disc)

    for a in ax.ravel():
        a.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        a.axis("off")

    cbar0 = fig.colorbar(im0, 
                        ax=ax[0], 
                        orientation='vertical', 
                        fraction=0.046, 
                        pad=0.04)
    cbar0.ax.tick_params(labelsize=22)

    cbar1 = fig.colorbar(im1, 
                        ax=ax[1], 
                        orientation='vertical', 
                        fraction=0.046, 
                        pad=0.04,
                        ticks=[disc_img_min, disc_img_max])
    cbar1.ax.tick_params(labelsize=22)
    # fig.tight_layout()
    if savefig:
        plt.savefig(os.path.join(base_dir,
                                 "ic_comparison.png"), dpi=200)
        plt.close()
    else:
        # return fig
        plt.show()
    
# %%

raw_final_ic_data = np.load("../data/ic_plot_sim_key_2134.npz")

raw_ic = raw_final_ic_data["nu_raw"]
final_ic = raw_final_ic_data["nu_final"]

show_raw_and_final_ic(raw_ic, final_ic, savefig=False)


# %%