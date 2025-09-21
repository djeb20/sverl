import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman"
})
textcolors = ("black", "white")

# For marking empty regions with darker boarders
def get_box(coord):
    
    ret = [np.array(coord) + [-0.5, -0.5]]
    ret.append(ret[-1] + [0, 1])
    ret.append(ret[-1] + [1, 0])
    ret.append(ret[-1] + [0, -1])
    ret.append(ret[-1] + [-1, 0])
    
    return ret

# Load in the Shapley values (get option actions for policy)
with open('PolicyShapley.pkl', 'rb') as file: 
    policy_sv = pickle.load(file)
    policy_sv = {state: values[:, action] for (state, values), action in zip(policy_sv.items(), [11, 3])}

with open('PerformanceShapley.pkl', 'rb') as file: 
    perf_sv = pickle.load(file)

with open('ValueShapley.pkl', 'rb') as file: 
    value_sv = pickle.load(file)

# Initialise figure
fig, axs = plt.subplots(2, 4)
fontsize = 12 # 15

# Loop over the different SVERL explanations (first is just game state)
for n, (row, sv, label) in enumerate(zip(
    axs.T,
    [{tuple(state): np.zeros(16) for state in policy_sv}, policy_sv, perf_sv, value_sv],
    ['Episode', 'Behaviour', 'Outcome', 'Prediction'])):

    # Loop over the different states
    for i, (ax, (state, shapley_values)) in enumerate(zip(row, sv.items())):

        # Reshape + scale Shapley values between -1 and 1 (if not all zero)
        shapley_values = shapley_values.reshape(4, 4)
        if shapley_values.max() - shapley_values.min() != 0:
            shapley_values = shapley_values / max(np.abs(shapley_values.max()), np.abs(shapley_values.min()))

        # Plot Shapley values
        im = ax.imshow(shapley_values, cmap='RdBu', norm=TwoSlopeNorm(0, -1, 1))

        # Reshape state and convert to readable format
        state = np.array(state, int).astype('O').reshape(4, 4)
        state[state == -1] = ' '
        state[state == 9] = 'M'

        # Loop over data dimensions and create text annotations.
        for k in range(len(state[0])):
            for j in range(len(state[1])):
                text = ax.text(j, k, state[k, j],
                    ha="center", va="center", 
                    color=textcolors[int(abs(shapley_values[k,j]) > 1/2)],
                    fontweight="bold", fontsize=fontsize)

        # Title on top axes and place mines for state.                                
        if i == 0: 
            ax.set_title(label, fontsize=fontsize) 
            ax.text(2, 2, 'M$_1$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
            ax.text(3, 1, 'M$_2$?', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
            ax.text(3, 0, 'M$_2$?', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)

        # Place mines for other state
        else:
            ax.text(2, 2, 'M$_1$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
            ax.text(3, 1, 'M$_2$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)

        # Place coordinates on the axes
        for j in range(4):

            # y-coordinates only on left column
            if n == 0:
                ax.text(-0.7, 3 - j, '{}'.format(j+1), ha="center", va="center", fontweight="bold", fontsize=fontsize)
                ax.text(-1, 1.5, '$y$', ha="center", va="center", fontweight="bold", fontsize=fontsize)
            
            # x-coordinates only on bottom row
            if i == 1:
                ax.text(j, 3.9, '{}'.format(j+1), ha="center", va="center", fontweight="bold", fontsize=fontsize)
                ax.text(1.5, 4.2, '$x$', ha="center", va="center", fontweight="bold", fontsize=fontsize)

        # Remove tickmarks
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        
        # Turn spines off and create white grid.
        ax.spines[:].set_linewidth(2)

        # Plotting bold lines around empty grid squares
        empty_coords = np.array(np.where(state == ' ')).T
        boxes = np.array([get_box(coord[::-1]) for coord in empty_coords])
        pairs = np.array([box[ind:ind+2] for box in boxes for ind in range(len(box[:-1]))])
        for _, box in enumerate(boxes):
            for ind in range(len(box[:-1])):
                coords = box[ind:ind+2]                
                if not (np.flipud(coords) == pairs).all(axis=(1, 2)).any():
                    ax.plot(coords.T[0], coords.T[1], c='k', lw=1.5)

        # Set ticks
        ax.set_xticks(np.arange(state.shape[1]-1)+.5, minor=False)
        ax.set_yticks(np.arange(state.shape[0]-1)+.5, minor=False)
        ax.grid(which="major", color='k', linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)

# Add colour bar
cbar = fig.colorbar(im, ax=axs, shrink=.7, fraction=0.1, orientation="vertical", anchor=(3.1, 0.5), pad=0, ticks=[-1, 0, 1])
cbar.ax.tick_params(labelsize=fontsize)

fig.supylabel('Shapley Value', x=1.08, fontsize=fontsize, rotation=-90, ha='right')

# Adjust layout and save figure
plt.subplots_adjust(bottom=-.65, left=-0.2)
plt.tight_layout()
plt.savefig('minesweeper_policy_perf_value.pdf', bbox_inches='tight', transparent=True)