import random
import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
import numpy as np

class FactoredTaxi(TaxiEnv):
    """
    Change Taxi state to have 4 features.
    """

    def __init__(self, seed:int=None):
        super().__init__()

        self.observation_space = gym.spaces.MultiDiscrete([5, 5, 5, 4])

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Generate transition dynamics with features.
        self.factored_P()

    def seed(self, seed:int):
        """
        Sets seed for environment.
        """

        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def factored_P(self):
        """
        Change transition dynamics to have 4 features.
        """

        self.P_hat = {tuple(self.decode(s)): {a: [[trans[0][0], np.array(list(self.decode(trans[0][1]))), *trans[0][2:]]] 
                                              for a, trans in values.items()} 
                                              for s, values in self.P.items()}

    def reset(self, seed:int=None):

        if seed is not None: 
            self.seed(seed)

        obs, info = super().reset()

        return np.array(list(self.decode(obs))), info
    
    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        return np.array(list(self.decode(obs))), reward, terminated, truncated, info
