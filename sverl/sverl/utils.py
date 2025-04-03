from collections import defaultdict
import numpy as np
from tqdm import tqdm

def train_agent(agent, env, total_timesteps, seed:int=None):
    """
    Q-Learning loop.
    """

    # Reset env; track return and average return
    obs, _ = env.reset(seed=seed)
    avg_r, r, count = 0, 0, 0

    pbar =  tqdm(range(total_timesteps), 'Q-Learning')
    for _ in pbar:

        # Epsilon-greedy action 
        action = agent.choose_action(obs)

        # Environment step.
        n_obs, reward, terminated, truncated, _ = env.step(action)
        r += reward

        # Train
        agent.update(obs, action, reward, n_obs, terminated)

        if terminated or truncated:

            # Update average return
            avg_r = avg_r * count + r
            count += 1
            avg_r /= count
            if count % 10_000 == 0:
                pbar.set_description(f'Q-Learning; Avg Return: {avg_r:0.5f}')

            # Reset
            obs, _ = env.reset()
            r = 0

        else:
            obs = n_obs

def get_steady_state(agent, env, steps):
    """
    Collects steps in environment to approximate steady-state.    
        Note: Steady-state will be off if episodes truncate.
    """

    steady_state = np.empty((steps, np.prod(env.observation_space.shape)))
    
    # Measuring convergence
    old_dist = [0]

    obs, _ = env.reset()
    count = 0

    pbar = tqdm(range(steps), f'Approximating Steady State.')
    for idx in pbar:

        # Don't collect terminal states.
        steady_state[idx] = obs
        
        # Usual step.
        obs, _, terminated, truncated, _ = env.step(agent.choose_action(obs, exp=False))

        # Display convergence rate
        if (idx + 1) % (steps / 10) == 0:
            dist = np.unique(steady_state[:idx], return_counts=True, axis=0)[1] / (idx + 1)
            diff = np.mean((old_dist - dist[:len(old_dist)]) ** 2)
            pbar.set_description(f'Approximating Steady State. Convergence: {diff:0.10f}')
            old_dist = dist

        if terminated or truncated:
            count += 1
            obs, _ = env.reset()

    return steady_state

def value_iteration(env, gamma, policy='greedy'):
    """
    Performs value iteration.
    """

    # Initialise value function
    V_table = defaultdict(float)

    # Extract transition dynamics (Taxi has a different name)
    P = env.P_hat if env.spec.id == 'FactoredTaxi-v3' else env.P

    # Helper function for p(s', r | s, a) * (r + (1 - term) * gamma * V(s'))
    q = lambda transition: transition[0] * (transition[2] + (1 - transition[3]) * gamma * V_table[*transition[1]])

    # Perform value iteration
    delta = float('inf')
    while delta > 1e-10:
        delta = 0.
        for s in P:
            v_prev = V_table[s]

            # Q(s, a) = sum_{s', r} [ p(s', r | s, a) * (r + (1 - term) * gamma * V(s')) ]
            Q_sa = np.array([np.sum([q(transition) 
                                     for transition in P[s][a]]) 
                                     for a in range(env.action_space.n)])
            
            # Greedy vs evaluting for general policy.
            if policy == 'greedy': 
                if hasattr(env, 'valid_actions'): # For environments with illegal moves.
                    V_table[s] = Q_sa[env.valid_actions[np.array(s).tobytes()]].max()
                else:
                    V_table[s] = Q_sa.max()
            else:
                V_table[s] = (policy(np.array(s)) * Q_sa).sum()

            delta = max(delta, abs(v_prev - V_table[s]))

    # Get Q table.
    return {tuple(s): np.array([np.sum([q(transition) 
                                        for transition in P[s][a]]) 
                                        for a in range(env.action_space.n)]) 
                                        for s in P}

def normalise_transitions(P):
    """
    Input: A dict of all possible transitions accounting for stochasticity.
    Computes the transition probabilities by counting and normalising their frequency.
    Returns the transition dynamics.
    """

    # Now we have a dict of all possible transitions, we need to normalise them.
    new_P = defaultdict(lambda: defaultdict(list))

    # Normalise transition probabilities
    for s, values in tqdm(P.items(), 'Normalising Transitions'):
        for a, transitions in values.items():

            # Count unique transitions to compute p(s', r | s, a)
            _, index, counts = np.unique([np.r_[*t] for t in transitions], axis=0, return_index=True, return_counts=True)
            
            # Normalise transition probabilities based on counts
            new_P[s][a] = [[count / counts.sum()] + transitions[i][1:] for i, count in zip(index, counts)]

    return new_P