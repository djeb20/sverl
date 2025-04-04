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
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman"
})

# Converting state in readable codes.
move_dict = {-1: ' ', 0: 'A', 1: 'B'}

textcolors = ("black", "white")

# Load in Shapley values
with open('PolicyShapley.pkl', 'rb') as file: 
    policy_sv = pickle.load(file)
    # Only chosen actions:
    policy_sv = {state: values[:, action] for action, (state, values) in zip([0, 2, 1], policy_sv.items())}

with open('PerformanceShapley.pkl', 'rb') as file: 
    perf_sv = pickle.load(file)

with open('ValueShapley.pkl', 'rb') as file: 
    value_sv = pickle.load(file)

# Initialise figure
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
fontsize = 20

# States to explain / be plotted.
states = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1],
          [0, 0, 0, 1, 2, 1, 0, 0, -1, -1, -1, -1]]

# Loop over the different SVERL explanations (first is just game state)
for ii, (row, sv, label) in enumerate(zip(
    axs.T,
    [{tuple(state): np.zeros(12) for state in states}, policy_sv, perf_sv, value_sv], 
    ['Episode', 'Policy', 'Performance', 'Value Estimation'])):

    # Loop over the different states
    for i, (ax, state) in enumerate(zip(row, states)):

        # Reshape Shapley values and account for numerical errors with steady-state approximation.
        shapley_values = np.flipud(np.reshape(sv[*state], (3, 4))).round(2) 
        
        # Normalise Shapley values between -1 and 1 (if not all zero)
        if shapley_values.max() - shapley_values.min() != 0:
            shapley_values = shapley_values / max(np.abs(shapley_values.max()), np.abs(shapley_values.min()))

        # Reshape state and change code indexes to readable ones.
        state = np.flipud(np.reshape(state, (3, 4))).astype(object)
        state[:, 1:-1] = np.reshape([move_dict[i] for i in state[:, 1:-1].flatten()], (3, 2))
        state[:, 0] = [' ' if value == -1 else value for value in state[:, 0]]
        state[:, -1] = [' ' if value == -1 else value for value in state[:, -1]]
        

        # Create the colourmap (between -1 and 1).
        im = ax.imshow(shapley_values, cmap='RdBu', norm=TwoSlopeNorm(0, -1, 1))

        # Loop over data dimensions and create text annotations.
        for k in range(len(state)):
            for j in range(len(state[0])):
                text = ax.text(j, k, state[k, j],
                    ha="center", va="center", 
                    color=textcolors[int(abs(shapley_values[k, j]) > 1/2)],
                    fontweight="bold", fontsize=fontsize)

        # Remove tickmarks
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
       
        # Turn spines off and create white grid.
        ax.spines[:].set_linewidth(2)

        # Set title, labels and ticks.
        ax.set_xticks(np.arange(state.shape[1]-1)+.5, minor=False)
        ax.set_yticks(np.arange(state.shape[0]-1)+.5, minor=False)
        ax.grid(which="major", color='k', linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", top=False,bottom=False,left=False,right=False)
        if i == 0:
            ax.set_title(label, fontsize=fontsize, pad=8)

# Add colour bar
cbar = fig.colorbar(im, ax=axs, shrink=1, fraction=0.1, orientation="vertical", anchor=(3.7, 0.5), pad=0)
cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=fontsize)

fig.supylabel('Shapley Value', x=1.09, fontsize=fontsize, rotation=-90, ha='right')

# Adjust layout and save figure
plt.subplots_adjust(bottom=-.1, left=.3)
plt.tight_layout()
plt.savefig('mastermind_policy_perf_value.pdf', bbox_inches='tight', transparent=True)