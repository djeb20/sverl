import random
import numpy as np
from collections import defaultdict
import gymnasium

"""
2 x 4 Gridworld
    - Random start states: (0, 0) and (1, 0).
    - Goal states: (0, 2) and (1, 2).
    - Impassible block: (0, 1).
    - Reward is -1 each step and +10 for reaching the goal.
"""

class Grid(gymnasium.Env):

    def __init__(self, seed:int=None):
        super().__init__()

        # 4 height and 2 width
        self.H = 4; self.W = 2
        self.grid = np.zeros((self.H, self.W))

        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.array([self.W, self.H]))
        self.action_space = gymnasium.spaces.Discrete(4)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Domain characteristics
        self.states = np.array([[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]])
        self.start_states = np.array([[0, 0], [1, 0]])
        self.actions = {0: [0, 1], 1: [1, 0],
                        2: [0, -1], 3: [-1, 0]}

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
        Defines the transition dynamics.
        """

        self.P = defaultdict(lambda: defaultdict(list))

        for state in self.states:
            for action_ind, action in self.actions.items():

                # Converts action int to vector, 'Take step'
                n_state = np.array(state) + action

                # Check if "hit wall"
                if not ((0 <= n_state[1] < self.H) and (0 <= n_state[0] < self.W)) or (n_state == [0, 1]).all(): 
                    n_state = np.array(state)
                
                if n_state[1] == self.H-1: # Reached goal
                    reward, terminated = 9, True 

                else: 
                    reward, terminated = -1, False

                # Update transition dictionary
                self.P[*state][action_ind].append([1, n_state, reward, terminated, False])

    def reset(self, seed:int=None):
        """
        Randomly places the agent in one of the bottom two squares.
        """

        if seed is not None: 
            self.seed(seed)

        self.pos = random.choice(self.start_states)

        return self.pos.copy(), {}

    def step(self, action):
        """
        Takes a step in the environment
        """

        # Deterministic env so no sampling needed (just take the first).
        self.pos, reward, terminated, truncated = self.P[*self.pos][action][0][1:]

        return self.pos.copy(), reward, terminated, truncated, {}
    
    def render(self):
        print(self.pos)