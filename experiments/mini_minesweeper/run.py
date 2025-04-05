from dataclasses import dataclass
import gymnasium as gym

import numpy as np
from sverl.agent import Agent
from sverl.envs.mini_minesweeper import MiniMinesweeper
from sverl.utils import get_steady_state, train_agent
from sverl.characteristics import PolicyCharacteristic, PerformanceCharacteristic, ValueCharacteristic
from sverl.shapley import PolicyShapley, PerformanceShapley, ValueShapley

@dataclass
class ExpArgs:
    env_id: str = 'MiniMinesweeper-v0'
    """The environment ID."""
    seed: int = 0
    """The random seed."""
    steady_state: int = 1_000_000
    """The number of state to approximate the steady state."""
    rollout_eps: int = 1_000
    """The number of episodes to rollout when approximating characteristic."""

@dataclass
class AgentArgs:
    total_timesteps: int = 1_000_000 
    """The number of timesteps to train the agent."""
    epsilon: float = 0.05
    """The exploration rate."""
    gamma: float = 1
    """The discount factor."""
    alpha: float = 0.2
    """The learning rate."""

@dataclass
class EnvArgs:
    pass

# First observation to explain.
e_ob1 = np.array([
    0.,  0.,  1., -1., 
    0.,  1.,  2., -1., 
    0.,  1., -1., -1., 
    0.,  1.,  1.,  1.
])

# Possible locations of the mines in the first observation.
mine_locs1 = np.array(
    [[
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]],
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0]]
    ])

# Second observation to explain.
e_ob2 = np.array([
    0.,  0.,  1., -1., 
    0.,  1.,  2., -1., 
    0.,  1., -1.,  2., 
    0.,  1.,  1.,  1.
])

# Possible locations of the mines in the second observation.
mine_locs2 = np.array(
    [[
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0]]
    ])

e_obs = np.array([e_ob1, e_ob2])
mine_locs = {tuple(e_ob1): mine_locs1, 
             tuple(e_ob2): mine_locs2}


if __name__ == '__main__':

    # Experiment setup
    exp_args = ExpArgs()
    agent_args = AgentArgs()
    env_args = EnvArgs()
    
    # Environment setup
    env = gym.make(exp_args.env_id, seed=exp_args.seed, **vars(env_args))

    # Agent setup
    agent = Agent(env, agent_args)
    train_agent(agent, env, total_timesteps=agent_args.total_timesteps, seed=exp_args.seed)

    # Steady-state
    steady_state = get_steady_state(agent, env, exp_args.steady_state)

    # --------------- Policy ---------------

    # Policy characteristic
    policy_char = PolicyCharacteristic(agent, env, steady_state)
    policy_char.get_exact(e_obs=e_obs)

    # Policy Shapley
    policy_shapley = PolicyShapley(policy_char)
    policy_shapley.get_exact(e_obs=e_obs)

    # --------------- Performance ---------------

    # Performance characteristic
    performance_char = PerformanceCharacteristic(agent, env, policy_char, steady_state)
    performance_char.get_exact_minesweeper(e_obs=e_obs, mine_locs=mine_locs, 
                                           rollout_eps=exp_args.rollout_eps)

    # Performance Shapley
    performance_shapley = PerformanceShapley(performance_char)
    performance_shapley.get_exact(e_obs=e_obs)

    # --------------- Value ---------------

    # Value characteristic
    value_char = ValueCharacteristic(agent, env, steady_state)
    value_char.get_exact(e_obs=e_obs)

    # Value Shapley
    value_shapley = ValueShapley(value_char)
    value_shapley.get_exact(e_obs=e_obs)