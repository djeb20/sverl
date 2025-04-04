from collections import defaultdict
import pickle
import random
import numpy as np
import gymnasium
from sverl.utils import normalise_transitions
from tqdm import tqdm

"""
Tic-Tac-Toe
    - Agent plays against a (stochastic) Minimax.
    - Usual rules.
    - Rewards: -1 lose, 0 draw and 1 win.
"""

class TicTacToe(gymnasium.Env):
    
    def __init__(self, seed:int=None):
        super().__init__()        
        
        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.full(9, 3))
        self.action_space = gymnasium.spaces.Discrete(9)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Domain characteristics
        self.rewards = {0: 0, 1: 1, 2: -1}
        self.actions = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2)
            }

        # Save previously calculated valid actions for boards.
        self.valid_actions = ValidDict()

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

        # Temporary dict to store transitions
        P = defaultdict(lambda: defaultdict(list))

        # Helper function to recursively step through environment
        def inner(obs, action, P):

            # Place piece
            n_obs = obs.copy()
            n_obs[self.actions[action]] = 1

            # See if game is finished and who won
            terminated, winner = self.won(n_obs)

            # Games ends in user win or draw, populate transition
            if terminated:
                P[*obs.flatten()][action].append([1., n_obs.flatten(), self.rewards[winner], True, False])
            
            # Game not over so Minimax plays
            else:

                # Loop over Minimax's chosen actions
                for mini_action in self.score(n_obs, 2)[1]:

                    # Play oppenent mark
                    mini_obs = n_obs.copy()
                    mini_obs[self.actions[mini_action]] = 2

                    # See if game is finished and who won
                    terminated, winner = self.won(mini_obs)
                    
                    # Completes the transition
                    trans = [1., mini_obs.flatten(), self.rewards[winner], terminated, False]
                    P[*obs.flatten()][action].append(trans)

                    # If game isn't over, agent to play.
                    if not terminated:
                        for n_action in self.valid_actions[mini_obs.tobytes()]:
                            inner(mini_obs.copy(), n_action, P)

        # Any initial action is optimal
        self.mini_start_states = []
        for action in tqdm(range(self.action_space.n), 'Generating P'):

            # Agent starts
            inner(np.zeros((3, 3)), action, P)
        
            # Minimax starts
            mini_obs = np.zeros((3, 3))
            mini_obs[self.actions[action]] = 2

            # Saving to reset environment to minimax starting.
            self.mini_start_states.append(mini_obs.flatten())

            # Agent responds to start
            for agent_action in self.valid_actions[mini_obs.tobytes()]:
                inner(mini_obs.copy(), agent_action, P)

        # Compute the transition probabilities
        self.P = normalise_transitions(P)

        # with open('P.pkl', 'wb') as f:
        #     pickle.dump(dict(self.P), f)

        # with open('P.pkl', 'rb') as f:
        #     self.P = pickle.load(f)
        
    def reset(self, seed:int=None):
        """
        Resets the environment for a new game.
        Starting player is chosen randomly
        """

        if seed is not None: 
            self.seed(seed)

        if random.random() < 0.5:
            self.board = np.zeros(9)
        else:
            self.board = random.choice(self.mini_start_states)
        
        return self.board.copy(), {}
        
    def step(self, action):
        """
        Agent places mark, either:
            - Game ends with draw or win
        Or:    
            - Minimax plays and:
            - Games ends with draw or loss,
            - or play continues 
        """

        # Check whether action is legal
        if action not in self.valid_actions[self.board.tobytes()]:
            raise ValueError(f"Invalid action: {action}; Valid actions: {self.valid_actions[self.board.tobytes()]}")

        # Next states, rewards and their probabilities
        all_trans = self.P[*self.board][action]
        ps = [row[0] for row in all_trans]

        # Stochastically selecting one transition
        self.board, reward, terminated, truncated = all_trans[np.random.choice(len(all_trans), p=ps)][1:]

        return self.board.copy(), reward, terminated, truncated, {}
        
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

    def score(self, state, player):
        """
        Given a board and player, recursively finds best moves for Minimax.
        """

        # Check if game is over and who won.
        terminated, winner = self.won(state)

        # Return score (who won) if terminated
        if terminated:
            return (winner + 1) % 3 - 1, None

        # Else continue game with other player.
        else:

            # Swap player andtrack scores of each branch
            n_player = player % 2 + 1
            legal_moves = self.valid_actions[state.tobytes()]
            scores = np.empty(len(legal_moves))

            for i, action in enumerate(legal_moves):

                # Player places mark
                new_state = state.copy()
                new_state[self.actions[action]] = player

                # New state needs new score
                scores[i] = self.score(new_state, n_player)[0]

            # Best move/score depends on player
            if player == 1: 
                best_score = max(scores)
            elif player == 2: 
                best_score = min(scores)

            best_moves = legal_moves[scores == best_score]
                        
            return best_score, best_moves

class ValidDict(dict):

    # Valid actions in given state.
    def valid_actions(self, state): 
        return (state == 0).nonzero()[0]
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key))
        self.__setitem__(key, val)
        
        return val   