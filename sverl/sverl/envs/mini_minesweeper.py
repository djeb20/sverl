import itertools
import pickle
import numpy as np
from collections import defaultdict
import random
import gymnasium
from sverl.utils import normalise_transitions
from tqdm import tqdm

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
        Dynamically generates the transition dynamics
        """

        # Temporary dict to store transitions
        P = defaultdict(lambda: defaultdict(list))

        # Helper function to recursively step through environment
        def inner(obs, action, P): 

            # Coordinates of square
            action_yx = self.actions[action]
            n_obs = obs.copy()

            # Hit a mine
            if self.mines[*action_yx] == 1:
                n_obs[*action_yx] = 9
                P[*obs.flatten()][action].append([1., n_obs.flatten(), -1, True, False])

            # Not a mine
            else:
                # Reveal square and adjacent squares if square is empty
                self.reveal(n_obs, *action_yx)

                # Gameover if all squares except mines are revealed.
                if (n_obs == -1).sum() == self.num_mines: 
                    P[*obs.flatten()][action].append([1., n_obs.flatten(), 0, True, False])
                
                # Else it continues
                else:

                    # This completes transition
                    P[*obs.flatten()][action].append([1., n_obs.flatten(), 0, False, False])
                    
                    # Loop overactions for next transitions
                    for action in self.valid_actions[n_obs.tobytes()]:
                        inner(n_obs, action, P)

        # Consider all possible mine locations.
        combs = itertools.combinations(range(self.height*self.width), self.num_mines)
        for mines in tqdm([np.isin(np.arange(self.height * self.width), c).astype(int) for c in combs], 'Generating P'):
            self.mines = mines.reshape(self.height, self.width)
            
            # Consider all actions from initial state
            for action in range(self.action_space.n):
                inner(np.full((self.height, self.width), -1, float), action, P)

        # Compute the transition probabilities
        self.P = normalise_transitions(P)

        with open('P.pkl', 'wb') as f:
            pickle.dump(dict(self.P), f)

        # with open('P.pkl', 'rb') as f:
        #     self.P = pickle.load(f)

    def reset(self, seed:int=None):
        """
        Resets environment for new game.
        Mines are placed randomly
        """

        if seed is not None: 
            self.seed(seed)

        # Initialise board
        self.board = np.full((self.height, self.width), -1, float)
        
        # Sample mine locations
        self.mines = np.zeros((self.height, self.width))
        self.mines.flat[np.random.choice(self.height * self.width, self.num_mines, replace=False)] = 1

        return self.board.flatten(), {}

    # def step(self, action):
    #     """
    #     Agent selects square to open.
    #     If square is a mine, game ends.
    #     If square is empty, reveal square and adjacent squares.
    #     If square is a number, reveal square.
    #     """

    #     # Check whether action is legal
    #     if action not in self.valid_actions[self.board.tobytes()]:
    #         raise ValueError(f"Invalid action: {action}; Valid actions: {self.valid_actions[self.board.tobytes()]}")
        
    #     # Next states, rewards and their probabilities
    #     all_trans = self.P[*self.board][action]
    #     ps = [row[0] for row in all_trans]

    #     # Stochastically selecting one transition
    #     self.board, reward, terminated, truncated = all_trans[np.random.choice(len(all_trans), p=ps)][1:]

    #     return self.board.copy(), reward, terminated, truncated, {}
    
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
            self.reveal(self.board, *action_yx)

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
    
# -------------------------- OLD CODE --------------------------

# P through step function

# def get_P(self):
    #     """
    #     Dynamically generates the transition dynamics
    #     """

    #     # Temporary dict to store transitions
    #     P = defaultdict(lambda: defaultdict(list))

    #     # Helper function to recursively step through environment
    #     def inner(env, obs, action, P): 

    #         n_obs, reward, terminated, truncated, _ = env.step(action)

    #         P[*obs][action].append([1., n_obs, reward, terminated, truncated])

    #         if not terminated: 
    #             for action in self.valid_actions[n_obs.tobytes()]:
    #                 inner(copy.deepcopy(env), n_obs, action, P)

    #     # Every trajectory for all mine positions.
    #     combs = itertools.combinations(range(self.height*self.width), self.num_mines)
    #     for mines in tqdm([np.isin(np.arange(self.height * self.width), c).astype(int) for c in combs], 'Generating P'):
    #         obs, _ = self.reset()
    #         self.mines = mines.reshape(self.height, self.width)
    #         for action in range(self.action_space.n):
    #             inner(copy.deepcopy(self), obs, action, P)

    #     # Compute the transition probabilities
    #     self.P = normalise_transitions(P)

# P with all the comments and debugging

# def get_P(self):
#         """
#         Dynamically generates the transition dynamics
#         """

#         # Temporary dict to store transitions
#         P = defaultdict(lambda: defaultdict(list))

#         # Helper function to recursively step through environment
#         def inner(obs, action, P): 
            
#             # if self.count == 1_000_000:
#             #     print(len(set(tuple(sublist) for sublist in self.all_actions)), len(self.all_actions))
#             #     raise ValueError('Count reached 5!')

#             # Coordinates of square
#             action_yx = self.actions[action]

#             # Hit a mine
#             if self.mines[*action_yx] == 1:
#                 n_obs = obs.copy()
#                 n_obs[*action_yx] = 9
#                 P[*obs.flatten()][action].append([1., n_obs.flatten(), -1, True, False])

#                 # print(f'Actions: {actions} \n')
#                 # print(n_obs.reshape(4, 4), '\n\n')
#                 # self.all_actions.append(actions)

#                 # print('Mine Hit \n\n')
#                 # self.count += 1

#             # Not a mine
#             else:
#                 # Reveal square and adjacent squares if square is empty
#                 n_obs = self.reveal(obs.copy(), *action_yx)

#                 # print(f'Actions: {actions} \n')
#                 # print(n_obs.reshape(4, 4), '\n\n')
#                 # self.all_actions.append(actions)

#                 # Gameover if all squares except mines are revealed.
#                 if (n_obs == -1).sum() == self.num_mines: 
#                     P[*obs.flatten()][action].append([1., n_obs.flatten(), 0, True, False])

#                     # print('Good Game \n\n')
#                     # self.count += 1
#                     # self.all_actions.append(actions)
                
#                 # Else it continues
#                 else:

#                     # This completes transition
#                     P[*obs.flatten()][action].append([1., n_obs.flatten(), 0, False, False])
                    
#                     # Loop overactions for next transitions
#                     for action in self.valid_actions[n_obs.tobytes()]:
#                         inner(n_obs.copy(), action, P)

#         # Consider all possible mine locations.
#         combs = itertools.combinations(range(self.height*self.width), self.num_mines)
#         for mines in tqdm([np.isin(np.arange(self.height * self.width), c).astype(int) for c in combs], 'Generating P'):
#             self.mines = mines.reshape(self.height, self.width)
            
#             # Consider all actions from initial state
#             for action in range(self.action_space.n):
#                 # self.count = 0
#                 inner(np.full((self.height, self.width), -1, float), action, P)

#                 # print(f'Mines: {self.mines}')
#                 # print(f'Action: {action}')
#                 # print(f'Count: {self.count} \n\n')

#         # self.mines = np.array([[1, 0, 0, 0], 
#         #                        [0, 0, 1, 0], 
#         #                        [0, 0, 0, 0], 
#         #                        [0, 0, 0, 0]])
        
#         # action = 1
#         # self.count = 0
#         # actions = []
#         # self.all_actions = []

#         # inner(np.full((self.height, self.width), -1, float), action, P, actions + [action])

#         # print(f'Mines: {self.mines}')
#         # print(f'Action: {action}')
#         # print(f'Count: {self.count} \n\n')

#         # print(len(set(tuple(sublist) for sublist in self.all_actions)), len(self.all_actions))

#         # Compute the transition probabilities
#         self.P = normalise_transitions(P)

#         with open('P.pkl', 'wb') as f:
#             pickle.dump(dict(self.P), f)