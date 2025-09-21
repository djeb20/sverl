import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman"
})

# Mapping from actions to marks
move_dict = {0: ' ', 1: 'X', 2: 'O'}
textcolors = ("black", "white")

# Initialise the figure
fig, axs = plt.subplots(1, 2, figsize=(5, 3))

# Set the range for the color map
range_values = 0.3
        
for ax, label, filename in zip(axs, ['Outcome', 'Prediction'], ['PerformanceShapley.pkl', 'ValueShapley.pkl']):

    # Load the data
    with open(filename, 'rb') as file:
        sv = pickle.load(file)
        
    # Reshape state and Shapley values
    state = np.array([move_dict[i] for i in list(sv)[0] if i in [0, 1, 2]]).reshape(3, 3)
    shapley_values = np.array(list(sv.values())[0]).reshape(3, 3)

    # Plot Shapley values as colour map
    im = ax.imshow(shapley_values, cmap='Blues', norm=TwoSlopeNorm(range_values / 2, 0, range_values))

    # Loop over data dimensions and create text annotations.
    for i in range(len(state[0])):
        for j in range(len(state[1])):
            text = ax.text(j, i, state[i, j],
                ha="center", va="center", 
                color=textcolors[int(abs(shapley_values[i,j]) > range_values/2)],
                fontweight="bold", fontsize=20)

    # Remove tickmarks
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # Set title, ticks and labels
    ax.set_title(label, fontsize=15)
    ax.set_xticks(np.arange(state.shape[1]-1)+.5, minor=False)
    ax.set_yticks(np.arange(state.shape[0]-1)+.5, minor=False)
    ax.grid(which="major", color='k', linestyle='-', linewidth=2)
    ax.tick_params(which="minor", top=False,bottom=False,left=False,right=False)

xlabel = fig.supxlabel("Shapley Value", fontsize=15)
xlabel.set_y(-0.04)

plt.tight_layout()

# Add colour bar
cbar = fig.colorbar(im, ax=axs, shrink=1, fraction=0.1, orientation="horizontal", pad=0.05)
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([0, .1, .2, .3])

# Save figure
plt.savefig('tic_tac_toe_perf_value.pdf', bbox_inches='tight', transparent=True)