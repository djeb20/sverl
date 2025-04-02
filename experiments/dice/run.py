from dataclasses import dataclass
import gymnasium as gym

import numpy as np
from sverl.agent import Agent
from sverl.envs.dice import Dice
from sverl.utils import value_iteration, get_steady_state
from sverl.characteristics import ValueCharacteristic
from sverl.shapley import ValueShapley

@dataclass
class ExpArgs:
    env_id: str = 'Dice-v0'
    """The environment ID."""
    seed: int = 0
    """The random seed."""

@dataclass
class AgentArgs:
    epsilon: float = 1 # Irrelevant because we use value iteration.
    """The exploration rate."""
    gamma: float = 1
    """The discount factor."""
    alpha: float = 0.2
    """The learning rate."""

@dataclass
class EnvArgs:
    pass

# States to explain
e_obs = np.array([[3, 6], [1, 1]])

if __name__ == '__main__':

    # Experiment setup
    exp_args = ExpArgs()
    agent_args = AgentArgs()
    env_args = EnvArgs()
    
    # Environment setup
    env = gym.make(exp_args.env_id, seed=exp_args.seed, **vars(env_args))

    # Agent setup
    agent = Agent(env, agent_args)
    agent.Q_table = value_iteration(env, agent_args.gamma)

    # Steady-state
    steady_state = get_steady_state(agent, env, 100_000) 

    # Characteristic
    value_char = ValueCharacteristic(agent, env, steady_state)
    value_char.get_exact(disp=False)

    # Shapley
    value_shapley = ValueShapley(value_char)
    value_shapley.get_exact(disp=False, e_obs=e_obs)