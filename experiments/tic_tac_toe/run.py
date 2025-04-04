from dataclasses import dataclass
import gymnasium as gym

import numpy as np
from sverl.agent import Agent
from sverl.envs.tic_tac_toe import TicTacToe
from sverl.utils import get_steady_state, value_iteration
from sverl.characteristics import PolicyCharacteristic, PerformanceCharacteristic, ValueCharacteristic
from sverl.shapley import PerformanceShapley, ValueShapley

@dataclass
class ExpArgs:
    env_id: str = 'TicTacToe-v0'
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
e_obs = np.array([[0., 0., 0., 0., 1., 0., 2., 0., 2.]])

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

    # --------------- Performance ---------------

    # Policy characteristic
    policy_char = PolicyCharacteristic(agent, env, steady_state)
    policy_char.get_exact()

    # Performance characteristic
    performance_char = PerformanceCharacteristic(agent, env, policy_char, steady_state)
    performance_char.get_exact(e_obs=e_obs)

    # Performance Shapley
    performance_shapley = PerformanceShapley(performance_char)
    performance_shapley.get_exact(e_obs=e_obs)

    # # --------------- Value ---------------

    # Value characteristic
    value_char = ValueCharacteristic(agent, env, steady_state)
    value_char.get_exact()

    # Value Shapley
    value_shapley = ValueShapley(value_char)
    value_shapley.get_exact(e_obs=e_obs)