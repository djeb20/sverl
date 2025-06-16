import numpy as np
import matplotlib.pyplot as plt
import pickle

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

# Bar width
width = 0.3

# Converts state to feature labels
feature_dic = {0: '$y$', 1: '$x$', 2: 'P', 3: 'D'}

# Load in Shapley values
with open('PolicyShapley.pkl', 'rb') as file:
    policy_sv = pickle.load(file)

with open('PerformanceShapley.pkl', 'rb') as file:
    perf_sv = pickle.load(file)
        
with open('ValueShapley.pkl', 'rb') as file:
    value_sv = pickle.load(file)

# Fixing formats (only chosen actions for policy)
policy_sv = {state: values[:, 0] for state, values in policy_sv.items()}
perf_sv = {state: values[:, 0] for state, values in perf_sv.items()}
value_sv = {state: values[:, 0] for state, values in value_sv.items()}

# Initialise figure
fig, axes = plt.subplots(2, 3, figsize=(8, 3), sharex=True) # 5
axes[1][0].sharey(axes[0][0])
axes[1][1].sharey(axes[0][1])
axes[1][2].sharey(axes[0][2])

# Loop over the different SVERL explanations.
for col, sv, title in zip(
    axes.T, 
    [policy_sv, perf_sv, value_sv],
    ['Policy', 'Performance', 'Prediction']
    ):

    # Loop over the different states and their Shapley values
    rects = []
    for i, (ax, (state, shapley_values), colour, label) in enumerate(zip(
        col, 
        sv.items(),
        ['C0', 'C3'],
        ['Top state', 'Bottom state']
        )):

        # Plot Shapley values
        rects.append(ax.bar(np.arange(4) / 2, shapley_values, width, label=label, color=colour))
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

        # Label features on bottom and title on top.
        if i:
            ax.set_xlabel("Features")
        else:
            ax.set_title(title)            

# Set labels, ticks and limits
fig.supylabel("Shapley Value")
axes[0][0].set_xticks(np.arange(4) / 2, feature_dic.values())
axes[0, 0].set_ylim(-.5, .7)
axes[0, 1].set_ylim(-4, 5)
axes[0, 2].set_ylim(-7.5, 6)

# Plot legend in bottom right
axes[1][2].legend(handles=rects, loc='lower right')

fig.tight_layout()

# Save figure
plt.savefig('taxi_policy_perf_value.pdf', bbox_inches='tight', transparent=True)