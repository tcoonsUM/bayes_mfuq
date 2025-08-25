# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import yaml
from unet_fno import FNO2d
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
plt.rc('axes.spines', top=True, right=True)
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.dpi': 200})
plt.rcParams.update({'lines.linewidth': 2.5}) 
plt.rcParams.update({'boxplot.boxprops.linewidth': 2.5})
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'grid.alpha': 0.5})
plt.rcParams.update({'grid.linestyle': '--'})
# plt.rcParams['text.usetex'] = True
plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({'font.family': 'serif'})

if os.environ['HOME'] == '/home/ajivani':
    user='Aniket'
else:
    user='Thomas'

if user=='Aniket':
    example_path = '/home/ajivani/Covuq/Paper01/'
    sys.path.insert(0, example_path)
else:
    example_path = '/Users/me-tcoons/Documents/GitHub/AdaptiveCov/Paper01/'
    sys.path.insert(0, example_path)

g = torch.Generator()
g.manual_seed(42)

# we will train the FNO from scratch using the separately created training set
# pretrained_model_path = "/nfs/turbo/coe-xhuan/ajivani/PDEBench/pdebench/models/fno/2D_DarcyFlow_beta1.0_Train_FNO.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

yaml_file = os.path.join(example_path, "config_2D_train.yaml")
with open(yaml_file, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

init_key_train = yaml_data["init_key"]
init_key_test = yaml_data["init_key_test"]
    
darcy_raw_data = np.load("./darcy_flow_surrogate_training_set_init_key_start_{}.npz".format(init_key_train))
N_SURR = darcy_raw_data["uu_all"].shape[0]

darcy_test_data = np.load("./darcy_flow_surrogate_test_set_init_key_start_{}.npz".format(init_key_test))
N_TEST = darcy_test_data["uu_all"].shape[0]

# Load grid from saved file.
grid_single = torch.Tensor(np.load("./grid_xy_darcy.npy")[0, :, :, :])

grid_train = grid_single.unsqueeze(0).repeat(N_SURR, 1, 1, 1)
grid_test = grid_single.unsqueeze(0).repeat(N_TEST, 1, 1, 1)

x_train_raw = torch.Tensor(darcy_raw_data["nu_all"]).unsqueeze(-1)
x_test_raw = torch.Tensor(darcy_test_data["nu_all"]).unsqueeze(-1)

y_train_raw = torch.Tensor(darcy_raw_data["uu_all"]).unsqueeze(-1)
y_test_raw = torch.Tensor(darcy_test_data["uu_all"]).unsqueeze(-1)
xx = torch.cat((x_test_raw, grid_test), dim=-1)

# %%
num_channels = yaml_data["num_channels"]
modes = yaml_data["modes"]
width = yaml_data["width"]
initial_step = yaml_data["initial_step"]

model = FNO2d(num_channels=num_channels,
              modes1=modes, 
              modes2=modes, 
              width=width, 
              initial_step=initial_step).to(device)

# pred_raw_sample = model(x_test_raw, grid_test)
pred_raw_sample = model(torch.cat((x_test_raw, grid_test), dim=-1))
data_by_split = {}
data_by_split['train'] = [x_train_raw, grid_train, y_train_raw]
data_by_split['test'] = [x_test_raw, grid_test, y_test_raw]

# flops_fno = FlopCountAnalysis(model, torch.cat((x_test_raw, grid_test), dim=-1)[:2, :, :, :])

batch_size = 16
batch_size_test = N_TEST
num_workers = 1

train_ds = TensorDataset(*data_by_split['train'])
test_ds  = TensorDataset(*data_by_split['test'])
train_dl = DataLoader(train_ds, 
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    generator=g,
                    shuffle=True)
test_dl = DataLoader(test_ds, 
                    batch_size=batch_size_test, 
                    num_workers=num_workers, 
                    generator=g,
                    shuffle=False)

# %%
learning_rate = 1e-3
scheduler_step = 5
scheduler_gamma = 0.5
n_epochs = 40
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
total_train_history = []
criterion = nn.L1Loss()
model_name = "fno"
os.makedirs(os.path.join(example_path, "checkpoints"), exist_ok=True)
checkpoint_path = os.path.join(example_path, "checkpoints", "{}_darcy_beta_timing_test.pt".format(model_name))
train_model = True
time_runs = True

if time_runs:
    start_time = time.perf_counter()

if train_model:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
        total_train_history = checkpoint["total_train_history"]
    else:
        starting_epoch = 0
    
    for epoch in range(starting_epoch, n_epochs):
        train_loss = 0.0
        model.train()
        for i, (xx, grid_xy, yy) in enumerate(train_dl):
            optimizer.zero_grad()
            features = torch.cat((xx, grid_xy), dim=-1)
            y_pred = model(features)
            loss = criterion(y_pred.squeeze(-1), yy.to(device))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print(f"Epoch {epoch+1}/{n_epochs} - Batch {(i + 1):04d} Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[-1]:.5f}")
        train_loss /= len(train_dl)

        total_train_history.append(train_loss)

        print(f"Saving model to {checkpoint_path}")
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'total_train_history': total_train_history
            }, checkpoint_path)

        scheduler.step()
else:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint["epoch"]
        total_train_history = checkpoint["total_train_history"]
    else:
        raise FileNotFoundError(f"The path '{checkpoint_path}' does not exist, set train model to True")

if time_runs:
    end_time = time.perf_counter()
    total_train_time = (end_time - start_time)

    print("Training time for {:02d} epochs: {:.3f} ms".format(n_epochs, total_train_time*1000))

# %%
model.eval()
with torch.no_grad():
    for i, (xx, grid_xy, yy_truth) in enumerate(test_dl):
        features = torch.cat((xx, grid_xy), dim=-1)
        y_pred_test = model(features)
        residual_test = torch.abs(y_pred_test.squeeze(-1) - yy_truth.to(device))
        # loss_test = criterion(y_pred.squeeze(-1), yy.to(device))

# %%    
def show_test_preds(imgs_pred, imgs_truth, img_min, img_max, savefig=False):
    if not isinstance(imgs_pred, list):
        imgs_pred = [imgs_pred]
        
    if not isinstance(imgs_pred, list):
        imgs_truth = [imgs_truth]
        
    fig, axs = plt.subplots(ncols=2, nrows=30, squeeze=False, figsize=(2, 10))   
    ref_ax = axs[0, 0]
        
    for i, (img_pred, img_truth) in enumerate(zip(imgs_pred, imgs_truth)):
        img_pred = img_pred.detach()
        img_pred_np = img_pred.cpu().numpy()
        
        img_truth = img_truth.detach()
        img_truth_np = img_truth.cpu().numpy()
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        imp = axs[i, 0].imshow(img_pred_np, cmap="viridis", vmin=img_min, vmax=img_max)
        imt = axs[i, 1].imshow(img_truth_np, cmap="viridis", vmin=img_min, vmax=img_max)
    
    if savefig:
        plt.savefig("panels_predictions_{}.png".format(model_name), dpi=200)
        plt.close()
    else:
        return fig


uu_min = min(y_pred_test.min().item(), yy_truth.min().item())
uu_max = max(y_pred_test.max().item(), yy_truth.max().item())

imgs_uu_pred = [img for img in y_pred_test.squeeze()]
imgs_uu_truth = [img for img in yy_truth.squeeze()]

show_test_preds(imgs_uu_pred, imgs_uu_truth, uu_min, uu_max)


# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters = {total_params}")

        # model.eval()
        # for i, (norm_locs, feat_params, targets) in enumerate(val_dl):
        #     phi_locs = generate_phi_batch(norm_locs.to(device), knots_res)
        #     feat_batch = torch.hstack((feat_params.to(device), phi_locs))
        #     y_pred = model(feat_batch)
        #     val_loss_iter = criterion(y_pred, targets.to(device))
        #     val_loss += val_loss_iter.item()
    
    
