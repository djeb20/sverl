import copy
from itertools import product
import numpy as np
from collections import defaultdict
import random
import gymnasium
from sverl.utils import normalise_transitions

"""
Mastermind
    - Hidden code randomly generated each episode.
    - Agent has to guess code.
    - Receives clues:(1) no. digits in right place, (2) no. right digits in wrong place.
    - Episode terminates if code guessed or runs out of guesses.
    - Reward is -1 a guess and + number of guesses if correct.
"""

class Mastermind(gymnasium.Env):

    def __init__(self, code_size:int=2, guesses:int=3, digits:int=2, seed:int=None):
        """
        Args:
            - code_size: length of code to guess
            - guesses: number of guesses
            - digits: Number of possible digits in a code (with replacement)
        """
        super().__init__()

        # Domain characteristics
        self.code_size = code_size
        self.guesses = guesses
        self.codes = np.array(list(product(range(digits), repeat=code_size)))

        # Define spaces (not strictly correct)
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.full(guesses * (code_size + 2), digits))
        self.action_space = gymnasium.spaces.Discrete(len(self.codes))

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # For clearer rendering: Function mapping a digit to a letter
        self.peg_to_letter = np.vectorize(lambda peg: chr(peg + 64) if peg > 0 else ' ')

        # Generate transition dynamics
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
        Dynamically creates transition dict.
        """

        # Temporary dict to store transitions
        P = defaultdict(lambda: defaultdict(list))

        # Helper function to recursively step through environment
        def inner(env, obs, action, P): 

            n_obs, reward, terminated, truncated, _ = env.step(action)

            P[*obs][action].append([1., n_obs, reward, terminated, truncated])

            if not terminated: 
                for action in range(env.action_space.n):
                    inner(copy.deepcopy(env), n_obs, action, P)

        # Every trajectory for all hidden codes.
        for code in self.codes:
            obs, _ = self.reset()
            self.code = code
            for action in range(self.action_space.n):
                inner(copy.deepcopy(self), obs, action, P)

        # Compute the transition probabilities
        self.P = normalise_transitions(P)

    def reset(self, seed:int=None):
        """
        Resets environment to empty grid with new code.
        """

        if seed is not None: 
            self.seed(seed)

        # Clear board and sample code
        self.board = np.full((self.guesses, self.code_size + 2), -1, dtype=int)
        self.code = random.choice(self.codes)

        # For tracking guesses
        self.guess = 0

        return self.board.flatten(), {}
    
    def step(self, action):
        """
        Step in the environment, places guess in next available row.
        """

        # The number of pegs exactly right and the number in the wrong position
        guess = self.codes[action]
        num_exact, num_close = self.get_clues(guess)

        # Place guess and clues on board
        self.board[self.guess, 1:-1] = guess
        self.board[self.guess, 0] = num_close
        self.board[self.guess, -1] = num_exact
        
        # Increment guess number
        self.guess += 1

        # Games finishes with correct guess
        if num_exact == self.code_size:
            return self.board.flatten(), 0, True, False, {}

        # Games finishes without correct guess
        elif self.guess == self.guesses: # Finished game with no win
            return self.board.flatten(), -1, True, False, {}
        
        # Game continues
        else:
            return self.board.flatten(), -1, False, False, {}
    
    def get_clues(self, guess):
        """
        Based on a guess and the current code returns:
            - The number of pegs in the exact right position.
            - The number of pegs that are right but in the wrong position.
        """

        # Copy of code to edit
        code = self.code.copy()

        # Right position           
        num_exact = (guess == code).sum()

        # Wrong position
        num_close = - num_exact
        for a in guess:
            if a in code:
                num_close += 1
                code[(code == a).argmax()] = -100 # Mark as used
        
        return num_exact, num_close
    
    def render(self):
        """
        Renders environment, expensive!
        """

        grid_render = self.grid.copy().astype(object)
        grid_render[:, 1:-1] = self.peg_to_letter(grid_render[:, 1:-1])
        print(np.flipud(grid_render), '\n')