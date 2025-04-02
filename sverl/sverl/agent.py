import pickle
import numpy as np
from collections import defaultdict
    
class Agent:
    """
    Usual Q learning agent.
    """
    
    def __init__(self, env, agent_args, actions=None):

        self.action_space = env.action_space
        
        # Hyperparameters
        self.args = agent_args
        self.epsilon = agent_args.epsilon
        
        # Q values
        self.Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        # actions is a dictionary of valid actions for each state.
        if actions is None:
            self.actions = defaultdict(lambda: np.arange(env.action_space.n))
        else:
            self.actions = actions

    def choose_action(self, obs, exp=True):

        # Valid actions
        actions = self.actions[obs.tobytes()]

        # Epsilon greedy exploration
        if np.random.rand() < self.epsilon and exp: 
            return np.random.choice(actions)
        else: 
            q_values = self.Q_table[*obs][actions]
            return np.random.choice(actions[q_values == q_values.max()])

    def V(self, obs):
        # Value over valid actions.
        return self.Q_table[*obs][self.actions[obs.tobytes()]].max(keepdims=True)
    
    def pi(self, obs):

        # Valid actions.
        actions = self.actions[obs.tobytes()]

        # Valid Q values; round to avoid inaccurate values.
        q_values = self.Q_table[*obs][actions].round(2)

        # Policy
        pi = np.zeros(self.action_space.n)
        pi[actions] = q_values == q_values.max() 

        return pi / pi.sum()

    def update(self, obs, action, reward, n_obs, terminated):
        """
        Q learning update. Only look ahead over available actions.
        """

        # Clause to stop error if valid "actions" is empty at end of episode.
        if terminated: 
            q_max = 0
        else: 
            q_max = self.Q_table[*n_obs][self.actions[n_obs.tobytes()]].max()
        
        # Usual update
        td_error = reward + self.args.gamma * q_max - self.Q_table[*obs][action]
        self.Q_table[*obs][action] += self.args.alpha * td_error