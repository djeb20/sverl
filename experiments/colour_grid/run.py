from dataclasses import dataclass
import gymnasium as gym

from sverl.agent import Agent
from sverl.envs.colour_grid import ColourGrid
from sverl.utils import value_iteration
from sverl.characteristics import PolicyCharacteristic
from sverl.shapley import PolicyShapley

@dataclass
class ExpArgs:
    env_id: str = 'ColourGrid-v0'
    """The environment ID."""
    seed: int = 0
    """The random seed."""

@dataclass
class AgentArgs:
    epsilon: float = 1 # Irrelevant because we use value iteration.
    """The exploration rate."""
    gamma: float = .9 # Infinite horizon MDP.
    """The discount factor."""
    alpha: float = 0.2
    """The learning rate."""

@dataclass
class EnvArgs:
    pass

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

    # Steady-state: uniform over (3, Green), (4, Green), (1, Red), (2, Blue)
    steady_state = [[3, 0], [4, 0], [1, 1], [2, 2]]

    # Policy characteristic setup
    policy_char = PolicyCharacteristic(agent, env, steady_state)
    policy_char.get_exact()

    # Shapley setup
    policy_shapley = PolicyShapley(policy_char)
    policy_shapley.get_exact()