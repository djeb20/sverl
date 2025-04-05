import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

file_name = 'minesweeper_' + os.path.basename(os.path.abspath(__file__))[:-3] + '.pdf'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman"
})

move_dict = {i : i for i in range(8)}
move_dict[-1] = ' '
move_dict[9] = 'M'

fontsize = 15

with open('../local_sverl_pi_all.pkl', 'rb') as file: shap_values_dic = pickle.load(file)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 2]})

for i, ax in enumerate(axs[:, 0]):
    state = list(list(shap_values_dic)[i])
    state_p = np.array([move_dict[int(i)] for i in state]).reshape(4, 4)
    
    im = ax.imshow(np.zeros((4, 4)), cmap='RdBu', norm=TwoSlopeNorm(0, -.1, .1))

    # Loop over data dimensions and create text annotations.
    for k in range(len(state_p[0])):
        for j in range(len(state_p[1])):
            text = ax.text(j, k, state_p[k, j],
                ha="center", va="center", 
                color="black",
                fontweight="bold", fontsize=fontsize)
            
    ax.set_title('State {}'.format(i + 1), fontsize=20)
            
    if i == 0: 
        ax.text(2, 2, '\sc M$_1$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
        ax.text(3, 1, '\sc M$_2$?', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
        ax.text(3, 0, '\sc M$_2$?', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)

    elif i == 1:

        # Text annotations for bombs
        ax.text(2, 2, '\sc M$_1$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)
        ax.text(3, 1, '\sc M$_2$', ha="center", va="center", fontweight="bold", color='black', fontsize=fontsize)

    for j in range(4):
        ax.text(j, 3.9, '{}'.format(j+1), ha="center", va="center", fontweight="bold", fontsize=15)
        ax.text(-0.7, 3 - j, '{}'.format(j+1), ha="center", va="center", fontweight="bold", fontsize=15)

    ax.text(1.5, 4.2, '$x$', ha="center", va="center", fontweight="bold", fontsize=20)
    ax.text(-1, 1.5, '$y$', ha="center", va="center", fontweight="bold", fontsize=20)

    # Remove tickmarks
    ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    # Turn spines off and create white grid.
    ax.spines[:].set_linewidth(2)


    ax.set_xticks(np.arange(state_p.shape[1]-1)+.5, minor=False)
    ax.set_yticks(np.arange(state_p.shape[0]-1)+.5, minor=False)
    ax.grid(which="major", color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", top=False,bottom=False,left=False,right=False)

# ----------------------------------------------------------------------------------------

w = 0.2
bars = []

for ax, state_ind, actions, features, actions_coords, title, offsets in zip(axs[:, 1], [0, 1], 
                                                                   [[3, 7, 10, 11], [3, 7, 10]], 
                                                                   [[3, 7, 10, 11], [3, 7, 10]],
                                                                   [[(4, 4), (4, 3), (3, 2), (4, 2)], [(4, 4), (4, 3), (3, 2)]],
                                                                   ['State 1', 'State 2'],
                                                                           [[i * w for i in [-1.5, -0.5, 0.5, 1.5]], 
                                                                            [i * w for i in [-1, 0, 1]]]):

    shap_values_dic_ind = {key: value for i, (key, value) in enumerate(shap_values_dic.items()) if i == state_ind}

    # [feature, action]
    values = np.array([[value[feature][action] for action in actions] for feature in features for value in shap_values_dic_ind.values()])

    x_axs = np.arange(len(actions))

    for action, label in enumerate(['$\pi(s, a_{{{}{}}})$'.format(*action) for action in actions_coords]):
        
        bars.append(ax.bar(x_axs + offsets[action], values[:, action], label=label, width=w))

    if state_ind == 1: 
        ax.set_xlabel('Feature', fontsize=18)     
        ax.set_ylim(-.1, .1)
        ax.set_yticks([-.1, 0, .1])
    else: 
        ax.set_title('Explaining Policy', fontsize=19)
        ax.legend(fontsize=17, ncol=2, loc='upper left')
        
    ax.set_ylabel('Shapley Value', fontsize=19)
    ax.set_xticks(x_axs, actions_coords)
    # ax.set_title(title, fontsize=15)

    ax.tick_params(axis='both', labelsize=17)

    ax.axhline(0, color='grey', linewidth=0.5,linestyle='--')

plt.subplots_adjust(wspace=0.4)
plt.savefig(file_name, bbox_inches='tight', transparent=True)