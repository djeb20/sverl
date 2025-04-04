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
    steady_state: int = 1_000_000
    """The number of state to approximate the steady state."""

@dataclass
class AgentArgs:
    epsilon: float = 1 # Irrelevant for value iteration.
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
    steady_state = get_steady_state(agent, env, exp_args.steady_state)
    
    # Steady-state for explained states, to put in paper table.
    states, dist = np.unique(steady_state, axis=0, return_counts=True)
    print(f"Steady-state:\
          p({e_obs[0]}) = {dist[(e_obs[0] == states).all(axis=1)] / dist.sum()};\
          p({e_obs[1]}) = {dist[(e_obs[1] == states).all(axis=1)] / dist.sum()}")

    # Characteristic
    value_char = ValueCharacteristic(agent, env, steady_state)
    value_char.get_exact()

    # Shapley
    value_shapley = ValueShapley(value_char)
    value_shapley.get_exact(e_obs=e_obs)