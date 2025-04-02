from collections import defaultdict
import copy
import random
import numpy as np
import gymnasium

"""
Tic-Tac-Toe
    - Agent plays against a (stochastic) Minimax.
    - Usual rules.
    - Rewards: -1 lose, 0 draw and 1 win.
"""

class TicTacToe(gymnasium.Env):
    
    def __init__(self, seed:int=None):
        super().__init__()

        # Converts actions to grid coords
        self.actions = {0: (0, 0),
                        1: (0, 1),
                        2: (0, 2),
                        3: (1, 0),
                        4: (1, 1),
                        5: (1, 2),
                        6: (2, 0),
                        7: (2, 1),
                        8: (2, 2)}
        
        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.full(9, 3))
        self.action_space = gymnasium.spaces.Discrete(9)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Domain characteristics
        self.rewards = {0: 0, 1: 1, 2: -1}

        # Save previously calculated valid actions for boards.
        self.valid_dict = ValidDict()

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

        # Temporary dict to store transitions
        P = defaultdict(lambda: defaultdict(list))

        # Helper function to recursively step through environment
        def inner(board, action, P):

            # Place piece
            board[self.actions[action]] = 1

            # See if game is finished and who won
            terminated, winner = self.won(board)

            if terminated:
                reward = self.rewards[winner]
                # P = ...
                return None
            else:
                # If game is not over, computer plays.
                _, best_actions = self.score(board, 2)

                for action in best_actions:

                    n_board = board.copy()
                    n_board[self.actions[action]] = 2

                    # See if game is finished and who won
                    terminated, winner = self.won(board)
                    reward = self.rewards[winner]
                    # P = 

                    if terminated:
                        return None
                    else:
                        for action in self.valid_actions(n_board):
                            inner(n_board.copy(), action, P)

        board = np.zeros((3, 3), dtype=int)

        # Agent starts
        for action in self.valid_dict[board.tobytes()]:
            inner(board.copy(), action, P)
        
        # Minimax starts
        _, minimax_actions = self.score(board, 2)
        for action in minimax_actions:

            n_board = board.copy()
            n_board[self.actions[action]] = 2

            inner(n_board.copy(), action, P)

        # Normalise probabilities
        
        
    def reset(self):
        """
        Resets the environment for a new game.
        Or sets env to given state.
        """

        self.board = np.zeros((3, 3), dtype=int)
        if np.random.rand() < 0.5: 
            self.minmax_player() # Starting player is chosen randomly
        
        return self.board.flatten(), {'valid_actions': self.valid_dict[self.board.tobytes()]}
        
    def step(self, action):
        """
        Takes a step and returns reward etc.
        """

        if action in self.valid_dict[self.board.tobytes()]:

            # Place piece
            self.board[self.action_dict[action]] = 1

            # See if game is finished and who won
            done, winner = self.won_dict[self.board.tobytes()]

            if not done:

                # If game is not over, computer plays.
                self.minmax_player()

                done, winner = self.won_dict[self.board.tobytes()]

            reward = self.reward_dict[winner]
    
        else:

            raise ValueError(f"Invalid action: {action}. Valid actions are: {self.valid_dict[self.board.tobytes()]}")

        return self.board.flatten(), reward, done, False, {'valid_actions': self.valid_dict[self.board.tobytes()]}
        
    # Currently valid actions.
    def valid_actions(self, state): 
        return (state == 0).nonzero()[0]
        
    def won(self, state):
        """
        Checks if the game is over and whether it is a draw or who won.
        0 draw, 1 human win, 2 comp win
        """
        
        # Fetch all rows, cols ad diags
        lines = np.concatenate((state, state.T, np.diag(state).reshape(1, -1), np.diag(np.fliplr(state)).reshape(1, -1)), axis=0)

        # Gameover or not
        if (state == 0).any(): 
            terminated = False
        else:
            terminated = True

        for line in lines:

            # Human win
            if np.all(line == 1):
                return True, 1
            
            # Opponent win
            elif np.all(line == 2):
                return True, 2
            
        return terminated, 0
            
# THIS IS ALL FOR MINMAX

    def minmax_player(self):
        """
        A minmax player plays the optimal move.
        """

        _, best_moves = self.score_dict[tuple([self.board.tobytes(), 2])]
        self.board[self.action_dict[np.random.choice(best_moves)]] = 2

    def score(self, state, player):
        """
        Given the game state and whose turn it is returns a tuple (estimated game score, best move to play)
        """

        state_byte = state.tobytes()

        done, winner = self.won_dict[state_byte]

        if not done: end_score = None
        else: end_score = (winner + 1) % 3 - 1

        if end_score is not None: return end_score, None
        else:
            all_moves = self.valid_dict[state_byte]
                        
            scores = np.empty(len(all_moves))
            
            n_player = player % 2 + 1

            for i, action in enumerate(all_moves):

                new_state = state.copy()
                new_state[self.action_dict[action]] = player

                current_score, _ = self.score_dict[tuple([new_state.tobytes(), n_player])]

                scores[i] = current_score

            if player == 1: 
                best_score = max(scores)
            elif player == 2: 
                best_score = min(scores)

            best_moves = all_moves[scores == best_score]
                        
            return best_score, best_moves

class ValidDict(dict, TicTacToe):
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key, dtype=np.int_))
        self.__setitem__(key, val)
        
        return val   