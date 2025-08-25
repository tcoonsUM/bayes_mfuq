import os
import sys
import random
import copy
from math import ceil, exp, log
import yaml
import jax
from jax import vmap
import jax.numpy as jnp
from jax import device_put, lax
import numpy as np
import torchvision
import torchvision.transforms.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from darcy_flow_utils import Courant_diff_2D, bc_2D, init_multi_2DRand

plt.rc('axes.spines', top=True, right=True)
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.dpi': 200})
plt.rcParams.update({'lines.linewidth': 2.5}) 
plt.rcParams.update({'boxplot.boxprops.linewidth': 2.5})
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'grid.alpha': 0.5})
plt.rcParams.update({'grid.linestyle': '--'})
plt.style.use('seaborn-v0_8-bright')
# plt.rcParams.update({'font.family': 'serif'})


def _pass(carry):
    return carry


def run_hifi_darcy(cfg, 
                   nsims=1, 
                   init_key=2026,
                   box_only=False):
    """
    Accepts a dictionary cfg of the following parameters:
    nx, ny: Grid resolution
    xL, xR, yL, yR: Dimensions of spatial domain
    ini_time: Start of time-dependent simulation
    fin_time: Stopping time of time-dependent simulation
    show_steps:
    CFL: CFL number
    dt_save: save frequency of solution
    beta: value of forcing term, set to 1 in our case.
    init_key: setting seed parameter for generating IC
    Keyword params:
        nsims: Number of hifi sims to generate.
        box_only: Currently unused keyword param, decides whether to return scalar QoI from averaging over prespecified bounding box or full state.
    """
    # basic parameters
    dx = (cfg['xR'] - cfg['xL']) / cfg['nx']
    dx_inv = 1. / dx
    dy = (cfg['yR'] - cfg['yL']) / cfg['ny']
    dy_inv = 1./dy

    # cell edge coordinate
    xe = jnp.linspace(cfg['xL'], cfg['xR'], cfg['nx'] + 1)
    ye = jnp.linspace(cfg['yL'], cfg['yR'], cfg['ny'] + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy

    show_steps = cfg['show_steps']
    ini_time = cfg['ini_time']
    fin_time = cfg['fin_time']
    dt_save = cfg['dt_save']
    CFL = cfg['CFL']
    beta = cfg['beta']

    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u, nu):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.
        uu = jnp.zeros([it_tot, u.shape[0], u.shape[1]])
        uu = uu.at[0].set(u)

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _show(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu, nu = carry

            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

            carry = (u, t, dt, steps, tsave, nu)
            u, t, dt, steps, tsave, nu = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu, nu)

        carry = t, tsave, steps, i_save, dt, u, uu, nu
        t, tsave, steps, i_save, dt, u, uu, nu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        return uu

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave, nu = carry
        dt = Courant_diff_2D(dx, dy, nu) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt, nu = carry
            # predictor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5, nu)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt, nu)
            return u, dt, nu

        carry = u, dt, nu
        u, dt, nu = lax.cond(dt > 1.e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave, nu

    @jax.jit
    def update(u, u_tmp, dt, nu):
        # boundary condition
        _u = bc_2D(u_tmp, mode='Neumann')
        # diffusion
        dtdx = dt * dx_inv
        dtdy = dt * dy_inv
        fx = - 0.5 * (nu[2:-1, 2:-2] + nu[1:-2, 2:-2]) * dx_inv * (_u[2:-1, 2:-2] - _u[1:-2, 2:-2])
        fy = - 0.5 * (nu[2:-2, 2:-1] + nu[2:-2, 1:-2]) * dy_inv * (_u[2:-2, 2:-1] - _u[2:-2, 1:-2])
        u -= dtdx * (fx[1:, :] - fx[:-1, :])\
           + dtdy * (fy[:, 1:] - fy[:, :-1])
        # source term: f = 1 * beta
        u += dt * beta
        return u
    
    
    u = init_multi_2DRand(xc, yc, numbers=nsims, k_tot=4, init_key=init_key)
    u = device_put(u)  # putting variables in GPU (if available?)
    
    # generate random diffusion coefficient
    key = jax.random.PRNGKey(init_key)
    xms = jax.random.uniform(key, shape=[nsims, cfg['ncoords']], minval=cfg['xL'], maxval=cfg['xR'])
    key, subkey = jax.random.split(key)
    yms = jax.random.uniform(key, shape=[nsims, cfg['ncoords']], minval=cfg['yL'], maxval=cfg['yR'])

    print("Generated samples for mixture")
    key, subkey = jax.random.split(key)
    stds = 0.5*(cfg['xR'] - cfg['xL']) * jax.random.uniform(key, shape=[nsims, cfg['ncoords']])
    nu = jnp.zeros_like(u)
    for i in range(cfg['ncoords']):
        nu += jnp.exp(-((xc[None, :, None] - xms[:, None, None, i]) ** 2
                        + (yc[None, None, :] - yms[:, None, None, i]) ** 2) / stds[:, None, None, i])
    nu = jnp.where(nu > nu.mean(), 1, 0.1)
    nu = vmap(bc_2D, axis_name='i')(nu)

    local_devices = jax.local_device_count()
    print("Initialized solution and IC")
    if local_devices > 1:
        nb, nx, ny = u.shape
        vm_evolve = jax.pmap(jax.vmap(evolve, axis_name='j'), axis_name='i')
        uu = vm_evolve(u.reshape([local_devices, nsims//local_devices, nx, ny]),\
                      nu.reshape([local_devices, nsims//local_devices, nx+4, ny+4]))
        uu = uu.reshape([nb, -1, nx, ny])
    else:
        vm_evolve = vmap(evolve, 0, 0)
        uu = vm_evolve(u, nu)

    # return np.array(xc), np.array(yc), np.array(tc), np.array(nu[:, 2:-2, 2:-2]), np.array(uu[:, -1, :, :])
    return np.array(nu[:, 2:-2, 2:-2]), np.array(uu[:, -1, :, :])

def generate_ic_from_z_samps(z_samps, cfg, seed=42, return_raw=False):
    nsims = z_samps.shape[1]
    # basic parameters
    dx = (cfg['xR'] - cfg['xL']) / cfg['nx']
    dx_inv = 1. / dx
    dy = (cfg['yR'] - cfg['yL']) / cfg['ny']
    dy_inv = 1./dy

    # cell edge coordinate
    xe = jnp.linspace(cfg['xL'], cfg['xR'], cfg['nx'] + 1)
    ye = jnp.linspace(cfg['yL'], cfg['yR'], cfg['ny'] + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy

    show_steps = cfg['show_steps']
    ini_time = cfg['ini_time']
    fin_time = cfg['fin_time']
    dt_save = cfg['dt_save']
    CFL = cfg['CFL']
    beta = cfg['beta']

    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    u = init_multi_2DRand(xc, yc, numbers=nsims, k_tot=4, init_key=seed)
    u = device_put(u)  # putting variables in GPU (if available?)
    
    print("Generated samples for mixture")
    
    # split z_samps into 3 chunks of size nsims x K representing xms, yms, stds
    xms, yms, stds = np.split(z_samps, 3)
    
    xms = jnp.array(xms.T)
    yms = jnp.array(yms.T)
    stds = jnp.array(stds.T)

    nu = jnp.zeros_like(u)
    for i in range(cfg['ncoords']):
        nu += jnp.exp(-((xc[None, :, None] - xms[:, None, None, i]) ** 2
                        + (yc[None, None, :] - yms[:, None, None, i]) ** 2) / stds[:, None, None, i])
    if return_raw:
        nu_raw = np.array(nu)
    
    nu = jnp.where(nu > nu.mean(), 1, 0.1)
    nu = vmap(bc_2D, axis_name='i')(nu)

    if return_raw:
        return nu_raw, np.array(nu[:, 2:-2, 2:-2])
    else:
        return np.array(nu[:, 2:-2, 2:-2])


def generate_ic_and_sol_from_z_samps(z_samps, cfg, seed=42):
    """
    Given an array of size (NX + NY + NSTD) x NSAMPLES, generate and return binary IC field and evolve the corresponding hifi solutions.
    """
    nsims = z_samps.shape[1]
    # basic parameters
    dx = (cfg['xR'] - cfg['xL']) / cfg['nx']
    dx_inv = 1. / dx
    dy = (cfg['yR'] - cfg['yL']) / cfg['ny']
    dy_inv = 1./dy

    # cell edge coordinate
    xe = jnp.linspace(cfg['xL'], cfg['xR'], cfg['nx'] + 1)
    ye = jnp.linspace(cfg['yL'], cfg['yR'], cfg['ny'] + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy

    show_steps = cfg['show_steps']
    ini_time = cfg['ini_time']
    fin_time = cfg['fin_time']
    dt_save = cfg['dt_save']
    CFL = cfg['CFL']
    beta = cfg['beta']

    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u, nu):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.
        uu = jnp.zeros([it_tot, u.shape[0], u.shape[1]])
        uu = uu.at[0].set(u)

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _show(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu, nu = carry

            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

            carry = (u, t, dt, steps, tsave, nu)
            u, t, dt, steps, tsave, nu = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu, nu)

        carry = t, tsave, steps, i_save, dt, u, uu, nu
        t, tsave, steps, i_save, dt, u, uu, nu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        return uu

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave, nu = carry
        dt = Courant_diff_2D(dx, dy, nu) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt, nu = carry
            # predictor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5, nu)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt, nu)
            return u, dt, nu

        carry = u, dt, nu
        u, dt, nu = lax.cond(dt > 1.e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave, nu

    @jax.jit
    def update(u, u_tmp, dt, nu):
        # boundary condition
        _u = bc_2D(u_tmp, mode='Neumann')
        # diffusion
        dtdx = dt * dx_inv
        dtdy = dt * dy_inv
        fx = - 0.5 * (nu[2:-1, 2:-2] + nu[1:-2, 2:-2]) * dx_inv * (_u[2:-1, 2:-2] - _u[1:-2, 2:-2])
        fy = - 0.5 * (nu[2:-2, 2:-1] + nu[2:-2, 1:-2]) * dy_inv * (_u[2:-2, 2:-1] - _u[2:-2, 1:-2])
        u -= dtdx * (fx[1:, :] - fx[:-1, :])\
           + dtdy * (fy[:, 1:] - fy[:, :-1])
        # source term: f = 1 * beta
        u += dt * beta
        return u
    
    
    u = init_multi_2DRand(xc, yc, numbers=nsims, k_tot=4, init_key=seed)
    u = device_put(u)  # putting variables in GPU (if available?)
    
    print("Generated samples for mixture")
    
    # split z_samps into 3 chunks of size nsims x K representing xms, yms, stds
    xms, yms, stds = np.split(z_samps, 3)
    
    xms = jnp.array(xms.T)
    yms = jnp.array(yms.T)
    stds = jnp.array(stds.T)

    nu = jnp.zeros_like(u)
    for i in range(cfg['ncoords']):
        nu += jnp.exp(-((xc[None, :, None] - xms[:, None, None, i]) ** 2
                        + (yc[None, None, :] - yms[:, None, None, i]) ** 2) / stds[:, None, None, i])
    nu = jnp.where(nu > nu.mean(), 1, 0.1)
    nu = vmap(bc_2D, axis_name='i')(nu)
    

    local_devices = jax.local_device_count()
    print("Initialized solution and IC")
    if local_devices > 1:
        nb, nx, ny = u.shape
        vm_evolve = jax.pmap(jax.vmap(evolve, axis_name='j'), axis_name='i')
        uu = vm_evolve(u.reshape([local_devices, nsims//local_devices, nx, ny]),\
                      nu.reshape([local_devices, nsims//local_devices, nx+4, ny+4]))
        uu = uu.reshape([nb, -1, nx, ny])
    else:
        vm_evolve = vmap(evolve, 0, 0)
        uu = vm_evolve(u, nu)

    return np.array(nu[:, 2:-2, 2:-2]), np.array(uu[:, -1, :, :])
    
    

def get_qoi_from_bounding_box(full_state,
                              yaml_data,
                              x_min=0.3,
                              x_max=0.45,
                              y_min=0.8,
                              y_max=0.95, 
                              qoi="mean"):
    """
    Returns scaler qoi (for now implemented qoi is mean of values in bounding box) given the full state array, xbounds and ybounds respectively.
    Size of full_state array: NSIMS x NX x NY
    Sample code for generating xbounds and ybounds:
    ny, nx = yaml_data["ny"], yaml_data["nx"]
    x_min_px, x_max_px = int(x_min * nx), int(x_max * nx)
    y_min_px, y_max_px = int(y_min * ny), int(y_max * ny)

    xbounds = (int(x_min * nx), int(x_max * nx))
    ybounds = (int(y_min * ny), int(y_max * ny))
    """

    ny, nx = yaml_data["ny"], yaml_data["nx"]
    x_min_px, x_max_px = int(x_min * nx), int(x_max * nx)
    y_min_px, y_max_px = int(y_min * ny), int(y_max * ny)

    xbounds = (int(x_min * nx), int(x_max * nx))
    ybounds = (int(y_min * ny), int(y_max * ny))

    assert full_state.shape[1] == yaml_data["nx"]
    assert full_state.shape[2] == yaml_data["ny"]
    bound_state = full_state[:, xbounds[0]:xbounds[1], ybounds[0]:ybounds[1]]

    if qoi == "mean":
        return torch.mean(bound_state, (1, 2)) - torch.mean(full_state, (1, 2))
    elif qoi == "quantile":
        # reshape to (NSIMS, bounds_x * bounds_y)
        bound_state_rs = bound_state.reshape(bound_state.shape[0], -1)
        return torch.quantile(bound_state_rs, 0.5, dim=1)
        # return torch.quantile(bound_state, 0.75, dim=(1, 2))

def get_point_qoi(full_state,
                yaml_data,
                x_q=0.3,
                y_q=0.6,
                ):
    """
    Returns a scaler point qoi given the full state array and desired location in the domain (x_q, y_q).
    If yaml_data["use_abs_qoi"] is set to False, returns a relative value of a point qoi contrasted with the mean surrogate solution. This is to eliminate potential bias when comparing model outputs.
    Size of full_state array: NSIMS x NX x NY
    """

    ny, nx = yaml_data["ny"], yaml_data["nx"]

    x_px = int(x_q * nx)
    y_px = int(y_q * ny)

    point_qoi = full_state[:, x_px, y_px]

    if yaml_data["use_abs_qoi"]:
        return point_qoi
    else:
        return point_qoi - torch.mean(full_state, (1, 2))

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
        return fig


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
        return fig

def show_raw_and_final_ic(raw_ic, final_ic, savefig=False, base_dir="/home/ajivani/Covuq/Paper01"):
    """
    filepath: "/home/ajivani/Covuq/Paper01/ic_plot_sim_key_2134.npz"
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    disc_img_min = 0.1
    disc_img_max = 1.0
    boundaries = [disc_img_min - 0.5, disc_img_min + 0.5, disc_img_max + 0.5]
    norm_disc = BoundaryNorm(boundaries, ncolors=2)
    base_cmap = plt.get_cmap('viridis')
    colors = [base_cmap(0.0), base_cmap(1.0)]
    discrete_cmap = ListedColormap(colors)
    # for i, (img, ax) in enumerate(zip(imgs, axs.ravel())):
    # img = img.detach()
    # img_np = img.cpu().numpy()
    
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

    if savefig:
        plt.savefig(os.path.join(base_dir,
                                 "ic_comparison.png"), dpi=200)
        plt.close()
    else:
        return fig