import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from sverl.agent import Agent
from sverl.envs.dice import Dice
from sverl.utils import value_iteration

class AgentArgs:
    epsilon: float = 1
    """The exploration rate."""
    gamma: float = 1
    """The discount factor."""

# Env
env = gym.make('Dice-v0')

# Agent setup
agent_args = AgentArgs()
agent = Agent(env, agent_args)
agent.Q_table = value_iteration(env, gamma=agent_args.gamma)

# Load in Shapley values
with open("ValueShapley.pkl", "rb") as f:
    shap = pickle.load(f)

# Shapley values
values = np.array(list(shap.values()))

# Optimal actions
actions = np.array([agent.choose_action(np.array(obs), exp=False) for obs in shap.keys()])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as necessary

# Iterate over each action to plot corresponding points
for action, (mark, label) in enumerate(zip(
    ['o', 's', 'D', '^'], 
    ['No dice', 'Dice 2', 'Dice 1', 'Both dice'])):
    
    v_values = values[actions == action]
    ax.scatter(v_values[:, 0], v_values[:, 1], marker=mark, s=90, edgecolor='w', label=label)

# Move the x and y axis to the center
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)

# Customize the tick marks
ax.set_xticks([-.3, 0.4])
ax.set_yticks([-.3, 0.4])

# Remove the top and right spines (the box around the plot)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Move the left and bottom spines to the zero point
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# Adjust the axis labels using labelpad and alignment to avoid crossing
ax.set_xlabel('Shapley Value for Dice 1', labelpad=20, ha='center')  # ha is for horizontal alignment
ax.set_ylabel('Shapley Value for Dice 2', labelpad=20, va='center')  # va is for vertical alignment

ax.xaxis.set_label_coords(0.64, 0.37)
ax.yaxis.set_label_coords(0.37, 0.64)

# Add legend
ax.legend(title="Re-Roll:", loc='upper right')

# Save figure
plt.savefig('dice.pdf', dpi=300, bbox_inches='tight')
