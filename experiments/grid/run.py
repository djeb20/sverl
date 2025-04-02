from dataclasses import dataclass
import gymnasium as gym

from sverl.agent import Agent
from sverl.envs.grid import Grid
from sverl.utils import value_iteration
from sverl.characteristics import PerformanceCharacteristic, PolicyCharacteristic
from sverl.shapley import PerformanceShapley

@dataclass
class ExpArgs:
    env_id: str = 'Grid-v0'
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

    # Steady-state: p(0, 0) = 1/7, p(rest) = 2/7
    steady_state = [[0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 2], [1, 2]]

    # Policy characteristic setup
    policy_char = PolicyCharacteristic(agent, env, steady_state)
    policy_char.get_exact(disp=False)

    # Characteristic setup
    performance_char = PerformanceCharacteristic(agent, env, policy_char, steady_state)
    performance_char.get_exact(disp=False)

    # Shapley setup
    performance_shapley = PerformanceShapley(performance_char)
    performance_shapley.get_exact(disp=False)