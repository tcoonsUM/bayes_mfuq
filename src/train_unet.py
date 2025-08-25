import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
from unet_fno import UNet2d
from fvcore.nn import FlopCountAnalysis
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
    example_path = '/home/ajivani/AdaptiveCov/Paper01/'
    sys.path.insert(0, example_path)
else:
    example_path = '/Users/me-tcoons/Documents/GitHub/AdaptiveCov/Paper01/'
    sys.path.insert(0, example_path)

g = torch.Generator() # we pass this to dataloaders to reproduce the shuffled datasets
g.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yaml_file = "./config_2D_train.yaml"
with open(yaml_file, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)


init_key_train = yaml_data["init_key"]
init_key_test = yaml_data["init_key_test"]
in_channels = yaml_data["in_channels"]
out_channels = yaml_data["out_channels"]


darcy_raw_data = np.load("./darcy_flow_surrogate_training_set_init_key_start_{}.npz".format(init_key_train))
N_SURR = darcy_raw_data["uu_all"].shape[0]

darcy_test_data = np.load("./darcy_flow_surrogate_test_set_init_key_start_{}.npz".format(init_key_test))
N_TEST = darcy_test_data["uu_all"].shape[0]

# Load grid from saved file.
grid_single = torch.Tensor(np.load("./grid_xy_darcy.npy")[0, :, :, :])

grid_train = grid_single.unsqueeze(0).repeat(N_SURR, 1, 1, 1)
grid_test = grid_single.unsqueeze(0).repeat(N_TEST, 1, 1, 1)

x_train_raw = torch.Tensor(darcy_raw_data["nu_all"])
x_test_raw = torch.Tensor(darcy_test_data["nu_all"])

y_train_raw = torch.Tensor(darcy_raw_data["uu_all"])
y_test_raw = torch.Tensor(darcy_test_data["uu_all"])

model = UNet2d(in_channels=in_channels,
               out_channels=out_channels).to(device)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters = {total_params}")

nu_test = torch.Tensor(darcy_test_data["nu_all"])
uu_test = torch.Tensor(darcy_test_data["uu_all"])
flops_unet = FlopCountAnalysis(model, nu_test[:2, :, :].unsqueeze(1))
pred_raw_sample = model(nu_test[:2, :, :].unsqueeze(1))

data_by_split = {}
data_by_split['train'] = [x_train_raw, y_train_raw]
data_by_split['test'] = [x_test_raw, y_test_raw]

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

learning_rate = 1e-3
scheduler_step = 5
scheduler_gamma = 0.5

n_epochs = 40
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

total_train_history = []
time_runs = True


criterion = nn.L1Loss()
model_name = "unet"
os.makedirs(os.path.join(example_path, "checkpoints"), exist_ok=True)
checkpoint_path = os.path.join(example_path, "checkpoints", "{}_darcy_beta_timing_test.pt".format(model_name))
train_model = True
if time_runs:
    start_time = time.perf_counter()

if train_model:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
    else:
        starting_epoch = 0
    
    for epoch in range(starting_epoch, n_epochs):
        train_loss = 0.0
        model.train()
        for i, (xx, yy) in enumerate(train_dl):
            optimizer.zero_grad()
            xx = xx.to(device)
            y_pred = model(xx.unsqueeze(1))
            loss = criterion(y_pred.squeeze(), yy.to(device))
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
            }, checkpoint_path)

        scheduler.step()
else:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        raise FileNotFoundError(f"The path '{checkpoint_path}' does not exist, set train model to True")

if time_runs:
    end_time = time.perf_counter()
    total_train_time = (end_time - start_time)

    print("Training time for {:02d} epochs: {:.3f} ms".format(n_epochs, total_train_time*1000))


        
model.eval()
with torch.no_grad():
    for i, (xx, yy_truth) in enumerate(test_dl):
        xx = xx.to(device)
        y_pred_test = model(xx.unsqueeze(1))
        residual_test = torch.abs(y_pred_test.squeeze() - yy_truth.to(device))

        
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

ft = show_test_preds(imgs_uu_pred, imgs_uu_truth, uu_min, uu_max)