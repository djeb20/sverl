import numpy as np
from collections import defaultdict
import random
import gymnasium

"""
Mini Minesweeper.
    - Usual Minesweeper rules.
    - Agent opens squares to reveal mines or information to find mines.
    - Reward is -1 if mine hit, 0 otherwise.
    - Edited from Scarllette Ellis's code.
"""

class MiniMinesweeper(gymnasium.Env):

    def __init__(self, width:int=4, height:int=4, num_mines:int=2, seed:int=None):
        super().__init__()

        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.full(height * width, 10))
        self.action_space = gymnasium.spaces.Discrete(height * width)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Domain characteristics
        self.height = height
        self.width = width
        self.num_mines = num_mines

        # Map discrete actions to coordinates
        self.actions = {
            action: (action // height, action % width) 
            for action in range(height * width)
        }

        # Save previously calculated valid actions for boards.
        self.valid_actions = ValidDict()

    def seed(self, seed:int):
        """
        Sets seed for environment.
        """

        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, seed:int=None):
        """
        Resets environment for new game.
        Mines are placed randomly
        """

        if seed is not None: 
            self.seed(seed)

        # Clear board
        self.board = np.full((self.height, self.width), -1, float)
        
        # Sample mine locations
        self.mines = np.zeros((self.height, self.width))
        self.mines.flat[np.random.choice(self.height * self.width, self.num_mines, replace=False)] = 1

        return self.board.flatten(), {}

    def step(self, action):
        """
        Agent selects square to open.
        If square is a mine, game ends.
        If square is empty, reveal square and adjacent squares.
        If square is a number, reveal square.
        """

        # Check whether action is legal
        if action not in self.valid_actions[self.board.tobytes()]:
            raise ValueError(f"Invalid action: {action}; Valid actions: {self.valid_actions[self.board.tobytes()]}")
        
        # Coordinates of square
        action_yx = self.actions[action]

        # Hit a mine
        if self.mines[*action_yx] == 1:
            self.board[*action_yx] = 9
            return self.board.flatten(), -1, True, False, {}

        # Not a mine
        else:
            # Reveal square and adjacent squares if square is empty
            self.board = self.reveal(self.board.copy(), *action_yx)

            # Gameover if all squares except mines are revealed.
            if (self.board == -1).sum() == self.num_mines: 
                return self.board.flatten(), 0, True, False, {}
            
            # Else it continues
            else:
                return self.board.flatten(), 0, False, False, {}
    
    def reveal(self, board, y, x):
        """
        Reveal square at (x, y) and adjacent squares if square is empty.
        """

        # Get slices for adjacent squares
        y0, y1 = max(0, y-1), min(y + 2, self.height)
        x0, x1 = max(0, x-1), min(x + 2, self.width)

        # Get number of mines in adjacent squares
        value = self.mines[y0:y1, x0:x1].sum()
        board[y, x] = value

        # If square is empty, reveal adjacent squares
        if value == 0:
            for j in range(y0, y1):
                for i in range(x0, x1):
                    if (i, j) != (x, y) and board[j, i] == -1:                        
                        self.reveal(board, j, i)

        return board

    def render(self):

        print()
        for row in self.board:
            to_print = ""
            for elm in row:
                to_print += " "
                if elm == -1: 
                    to_print += 'B'
                else:
                    to_print += str(int(elm))
            print(to_print)
        print()

class ValidDict(dict):

    # Valid actions in given state.
    def valid_actions(self, state): 
        return (state == -1).nonzero()[0]
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key))
        self.__setitem__(key, val)
        
        return val