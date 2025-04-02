import random
import numpy as np
from itertools import product
from collections import defaultdict
import gymnasium

"""
6-sided Dice-rolling game
    - Agent chooses which combination of dice to re-roll.
    - The game terminates with probability 1/n each re-roll, where n is the expected number of steps.
    - Reward is 1 if the sum of the dice is greater than some goal, 0 otherwise.
"""

class Dice(gymnasium.Env):

    def __init__(self, num_dice:int=2, goal:int=10, num_steps:int=2, seed:int=None):
        super().__init__()

        # Game parameters
        self.num_dice = num_dice
        self.num_steps = num_steps
        self.goal = goal

        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.array([6] * num_dice))
        self.action_space = gymnasium.spaces.Discrete(2 ** num_dice)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # All possible rolls and replacements
        self.rolls = np.array(list(product(range(1, 7), repeat=num_dice)))
        self.actions = np.array(list(product([False, True], repeat=num_dice)))

        # Define transition dynamics
        self.get_P()

    def seed(self, seed:int):
        """
        Sets seed for environment.
        """

        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def get_P(self):
        """
        Returns transitoin dynamics.
        """

        self.P = defaultdict(lambda: defaultdict(list))

        # Loop over states, actions, new rolls and termination flag.
        for state in self.rolls:
            for action_ind, action in enumerate(self.actions):
                for roll in self.rolls:
                    for terminated in [False, True]:
                        
                        # In next state replaced dice are taken from the new roll.
                        n_state = np.array(state)
                        n_state[action] = roll[action]

                        # Reward is 1 if dice sum >= goal, 0 otherwise.
                        reward = float(terminated * (n_state.sum() >= self.goal))

                        # Transition probability accounts for terminatoin and possible replacements.
                        p_term = terminated * (1 / self.num_steps) + (1 - terminated) * (1 - 1 / self.num_steps)
                        p = p_term / (6 ** len(action))

                        self.P[*state][action_ind].append([p, n_state, reward, terminated, False])

    def reset(self):
        """
        Resets environment by rolling dice.
        """

        self.roll = self.rolls[np.random.choice(len(self.rolls))].copy()
        
        return self.roll.copy(), {}
    
    def step(self, action):
        """
        Agent selects which dice to reroll.
        """

        # Next states, rewards and their probabilities
        all_trans = self.P[*self.roll][action]
        ps = [row[0] for row in all_trans]

        # Stochastically selecting one transition
        self.roll, reward, terminated, truncated = all_trans[np.random.choice(len(all_trans), p=ps)][1:]

        return self.roll.copy(), reward, terminated, truncated, {}
    
    def render(self):
        """
        Renders environment by printing out current roll
        """

        print(self.roll)