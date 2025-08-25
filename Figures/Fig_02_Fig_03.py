import numpy as np
import matplotlib.pyplot as plt
import pickle
    
with open('../Data/gg_mvn_projections/all_n_pilots_projected.pickle', 'rb') as file:
    all_n_pilots_projected = pickle.load(file)
    
with open('../Data/gg_mvn_projections/l_bars_post.pickle', 'rb') as file:
    l_bars_post = pickle.load(file)
    
with open('../Data/gg_mvn_projections/all_projected_l_bars.pickle', 'rb') as file:
    all_projected_l_bars = pickle.load(file)
    
acc_losses_proj_gg_mvn = np.load('../Data/gg_mvn_projections/acc_losses_gg_mvn.npy')
cost_losses_proj_gg_mvn = np.load('../Data/gg_mvn_projections/cost_losses_gg_mvn.npy')


#%% cost/loss accuracy decomp
plt.figure()
n_pilots = np.arange(6,180,step=2)
# Convert to numpy arrays for stacking
acc = np.array(acc_losses_proj_gg_mvn)
cost = np.array(cost_losses_proj_gg_mvn)

# Plot filled areas
plt.fill_between(n_pilots, 0, acc, label=r'Accuracy Loss $\overline{\mathcal{L}}_A$', color='purple', alpha=0.5)
plt.fill_between(n_pilots, acc, acc + cost, label=r'Cost Loss $\overline{\mathcal{L}}_C$', color='orange', alpha=0.5)

# Overlay the total loss as a dashed line
plt.plot(n_pilots, acc + cost, color='black', label=r'Total Loss $\overline{\mathcal{L}}$', linestyle='--')

total_loss = acc + cost
first_x = n_pilots[0]
first_y = total_loss[0]
plt.scatter([first_x], [first_y], color='black', s=50, facecolors='black', edgecolors='black', label=r'Initial $N_{\text{pilot}}$')
plt.text(first_x+1, first_y, ' 6', va='bottom', ha='left', fontsize=10)
plt.legend()
plt.xlabel(r'$N_{\text{pilot}}$')
plt.ylabel('Expected Loss')
plt.yscale('log')


#%% plot results
from matplotlib.lines import Line2D
n_pilots = np.arange(6, 100)
cmap = plt.cm.viridis_r#plt.cm.Blues
colors = cmap(np.linspace(0.3, .9, len(all_projected_l_bars)))  # adjust 0.3 to 0.9 for light to dark

plt.figure()

# Actual curve
plt.plot(n_pilots, l_bars_post, zorder=1, linewidth=2.5, color='black')

# Projected curves and markers
for i in range(len(all_projected_l_bars)):
    x0=all_n_pilots_projected[i][0]
    y0=all_projected_l_bars[i][0]
    plt.plot(
        all_n_pilots_projected[i],
        all_projected_l_bars[i],
        zorder=0,
        color=colors[i],
        linewidth=1.5,
        linestyle='dashed'
    )
    plt.scatter(
        x0,
        y0,
        color=colors[i],
        edgecolor='black',
        s=40,
        zorder=2
    )
    
    # Add label slightly above the marker
    if i==0 or i==1:
        x0 = x0 + 3
        y0 = y0 / 1.07
    plt.text(
        x0,
        y0 * 1.06,  # adjust vertical position for clarity
        str(all_n_pilots_projected[i][0]),
        fontsize=8,
        ha='center',
        va='bottom'
    )

#plt.yscale('log')

# Custom legend entries
legend_elements = [
    Line2D([0], [0], color='black', lw=2.5, label='Actual'),
    Line2D([0], [0], color=plt.cm.Blues(0.8), lw=1.5, ls='dashed', label='Projections'),
    Line2D([0], [0], marker='o', color=plt.cm.viridis_r(0.6), label=r'Initial $N_{\text{pilot}}$',
           markerfacecolor=plt.cm.Blues(0.8), markeredgecolor='black', markersize=6, linestyle='None')
]

plt.legend(handles=legend_elements)
plt.xlabel(r'$N_{\text{pilot}}$')
plt.ylabel(r'Expected Loss')