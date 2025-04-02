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

        # Domain characteristics
        self.rewards = {0: 0, 1: 1, 2: -1}

        # Save previously calculated properties of boards, i.e. if finished and who's won - for speed.
        self.won_dict = WonDict()

        # Save previously calculated valid actions for boards, also speed.
        self.valid_dict = ValidDict()

        # Save previous scores for minmax, speed.
        self.score_dict = ScoreDict()
        self.score_dict.won_dict = self.won_dict
        self.score_dict.valid_dict = self.valid_dict
        self.score_dict.action_dict = self.action_dict
        self.score_dict.score_dict = self.score_dict
        
    def reset(self, start_state=None):
        """
        Resets the environment for a new game.
        Or sets env to given state.
        """

        if start_state is None:

            self.board = np.zeros((3, 3), dtype=int)
            if np.random.rand() < 0.5: 
                self.minmax_player() # Starting player is chosen randomly

        else: self.board = start_state.copy().reshape(3, 3)
        
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

            reward = 0
            done = False

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

# These dictionaries are just for speeding up some stuff.

class WonDict(dict, TTT):
    
    def __missing__(self, key):
        
        val = self.won(np.frombuffer(key, dtype=np.int_).reshape(3, 3))
        self.__setitem__(key, val)
        
        return val

class ValidDict(dict, TTT):
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key, dtype=np.int_))
        self.__setitem__(key, val)
        
        return val

class ScoreDict(dict, TTT):
    
    def __missing__(self, key):
        
        state, player = key
        val = self.score(np.frombuffer(state, dtype=np.int_).reshape(3, 3), player)
        self.__setitem__(key, val)
        
        return val        