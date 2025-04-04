from collections import defaultdict
import pickle
from tqdm import tqdm
import numpy as np
from sverl.utils import value_iteration

class Characteristic:
    def __init__(self, agent, env, steady_state):

        self.agent = agent
        self.env = env

        # Number of features
        self.F_card = np.prod(env.observation_space.shape)

        # Buffer representative of steady state
        self.steady_state = steady_state

        # Mask for unknown features
        self.mask = -100
    
    def null(self, obs):
        # Null characteristic value.
        return self.get_exact_val(obs, np.zeros_like(obs))
    
    def all_C(self):
        # All coalitions
        return np.array(np.meshgrid(*[[0, 1]] * self.F_card)).T.reshape(-1, self.F_card)
    
# -------------------------------------------- VALUE AND POLICY CHARACTERISTICS --------------------------------------------

class ValuePolicyCharacteristic(Characteristic):
    def __init__(self, agent, env, steady_state):
        super().__init__(agent, env, steady_state)

    def get_exact(self, disp=False, save=True):
        """
        Calculates the exact characteristic values for policy and prediction.
            - Assumes steady_state is a good approximation.
        """

        # Approximate steady state.
        e_obs, dist = np.unique(self.steady_state, axis=0, return_counts=True)
        dist = dist / dist.sum()

        # Targets
        targets = np.array([self.v_F(e_ob) for e_ob in e_obs])

        self.exact = {}

        # Loop over coalitions
        for C in tqdm(self.all_C(), f'Exact {self.__class__.__name__}'):
            m_obs = np.where(C.astype(bool), e_obs, self.mask)

            # Loop over the unique partial observations
            for m_ob in np.unique(m_obs, axis=0):

                # Set the characteristic value to the average of the values of the matching observations.
                indexes = (m_ob == m_obs).all(axis=1)
                cond_dist = dist[indexes] / dist[indexes].sum()
                self.exact[*m_ob] = (targets[indexes] * cond_dist[:, None]).sum(axis=0)

        if save:
            with open(f'{self.__class__.__name__}.pkl', "wb") as f:
                pickle.dump(self.exact, f)

        if disp:
            print(f'Exact characteristic values: {self.exact}')

    def get_exact_val(self, e_ob, C):

        if tuple(np.where(C.astype(bool), e_ob, self.mask)) not in self.exact:
            print(f'm_ob not in exact; e_ob: {e_ob}, C: {C}, m_ob: {np.where(C.astype(bool), e_ob, self.mask)}')

        return self.exact[*np.where(C.astype(bool), e_ob, self.mask)].copy()
    
class ValueCharacteristic(ValuePolicyCharacteristic):
    def __init__(self, agent, env, steady_state):
        super().__init__(agent, env, steady_state)
        
        # Full characteristic value
        self.v_F = self.agent.V

class PolicyCharacteristic(ValuePolicyCharacteristic):
    def __init__(self, agent, env, steady_state):
        super().__init__(agent, env, steady_state)

        # Full characteristic value
        self.v_F = self.agent.pi

# -------------------------------------------- PERFORMANCE CHARACTERISTIC --------------------------------------------

class PerformanceCharacteristic(Characteristic):
    def __init__(self, agent, env, policy_char, steady_state):
        super().__init__(agent, env, steady_state)

        # Performance characteristics are calculated using the policy characteristic.
        self.char = policy_char

    def exact_pi(self, obs, e_obs, C):

        # Policy characteristic at explained state (e_obs)
        if (obs == e_obs).all():

            # Policy characteristic
            pi_C = self.char.get_exact_val(e_obs, C)

            # Mask out invalid actions
            valid_actions = self.agent.actions[e_obs.tobytes()]
            pi = np.zeros_like(pi_C, dtype=float)
            pi[valid_actions] = pi_C[valid_actions]

            # Normalise and return
            return pi / pi.sum()
        
        # Fully-observed pi elsewhere
        else:
            return self.char.v_F(obs)

    def choose_action(self, obs, e_obs, C):
        return np.random.choice(self.env.action_space.n, p=self.exact_pi(obs, e_obs, C))

    def get_exact(self, disp=False, e_obs=None, save=True):
        """
        Calculates the exact characteristic values for performance using value iteration.
        """

        # Get transition dicionary
        if not hasattr(self.env, 'P'):
            self.env.get_P()

        self.exact = defaultdict(dict)

        # Compute for all states unless given
        if e_obs is None:
            e_obs = np.unique(self.steady_state, axis=0)

        # Loop over all coalitions and states
        for C in tqdm(self.all_C(), f'Exact {self.__class__.__name__}'):
            for e_ob in e_obs:

                # Policy pi_hat from performance characteristic definition.
                policy = lambda obs: self.exact_pi(obs, e_ob, C)
                Q_table = value_iteration(self.env, gamma=self.agent.args.gamma, policy=policy)
                self.exact[*C][*e_ob] = (policy(e_ob) * Q_table[*e_ob]).sum(axis=0, keepdims=True)
                
        if save:
            with open(f'{self.__class__.__name__}.pkl', "wb") as f:
                pickle.dump(self.exact, f)

        if disp:
            print(f'Exact characteristic values: {self.exact}')

    def get_exact_val(self, e_obs, C):
        return self.exact[*C][*e_obs].copy()
    
    def get_exact_minesweeper(self, e_obs, mine_locs, rollout_eps, disp=False, save=True):
        """
        Calculates the exact characteristic values for performance using rollouts.
        Only for Minesweeper because env.P cannot be generated.
        Takes advantage of the domain being non-cyclic, meaning we only need to compute
        Q(s, a) for all a, then char = pi_C(s, a) * Q(s, a).
        """

        # First rollout original pi for all a from all states.
        Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Consider every state, action and possible mine locations. Average rollouts.
        for e_ob in e_obs:
            for a in tqdm(self.env.valid_actions[e_ob.tobytes()], 'Rollouts...'):
                for mines in mine_locs[*e_ob]:
                    for _ in range(rollout_eps // len(mine_locs[*e_ob])):

                        # Reset environment with mines and state
                        obs, _ = self.env.reset(options={'board': e_ob, 'mines': mines})
                        action = a; r = 0

                        while True:

                            # Usual RL, choose action, execute, update
                            obs, reward, terminated, truncated, _ = self.env.step(action)
                            r += reward

                            # If terminated, average reward over all rollouts
                            if terminated or truncated:

                                Q_table[*e_ob][a] += r / (len(mine_locs[*e_ob]) * (rollout_eps // len(mine_locs[*e_ob])))                                
                                break
                            
                            # Else, choose action from agent
                            else:
                                action = self.agent.choose_action(obs, exp=False)

        # Compute characteristic values
        self.exact = defaultdict(dict)

        # Loop over all coalitions and states
        for C in tqdm(self.all_C(), f'Exact {self.__class__.__name__}'):
            for e_ob in e_obs:

                # sum [ pi_C * Q(s, a) ]
                self.exact[*C][*e_ob] = (self.exact_pi(e_ob, e_ob, C) * Q_table[*e_ob]).sum(keepdims=True)
                
        if save:
            with open(f'{self.__class__.__name__}.pkl', "wb") as f:
                pickle.dump(self.exact, f)

        if disp:
            print(f'Exact characteristic values: {self.exact}')

# --------------------------------------------- OLD CODE --------------------------------------------

# def get_exact_minesweeper(self, e_obs, mine_locs, rollout_eps, disp=False, save=True):
#         """
#         Calculates the exact characteristic values for performance using rollouts.
#         Only for Minesweeper because env.P cannot be generated.
#         Takes advantage of the domain being non-cyclic, meaning we only need to compute
#         Q(s, a) for all a, then char = pi_C(s, a) * Q(s, a).
#         """

#         # First rollout original pi for all a from all states.
#         Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

#         # Consider every state, action and possible mine locations. Average rollouts.
#         for e_ob in e_obs:
#             # print(f'Explaining state: {e_ob}')
#             for a in tqdm(self.env.valid_actions[e_ob.tobytes()], 'Rollouts...'):
#                 # print(f'Action: {a}')
#                 for mines in mine_locs[*e_ob]:
#                     # print(f'Mines: {mines}')
#                     for _ in range(rollout_eps // len(mine_locs[*e_ob])):

#                         # Reset environment with mines and state
#                         obs, _ = self.env.reset(options={'board': e_ob, 'mines': mines})
#                         action = a; r = 0

#                         while True:

#                             # Usual RL, choose action, execute, update
#                             obs, reward, terminated, truncated, _ = self.env.step(action)
#                             r += reward

#                             if terminated or truncated:
#                                 # print(r)
#                                 break
#                             else:
#                                 action = self.agent.choose_action(obs, exp=False)

#                         Q_table[*e_ob][a] += r / (len(mine_locs[*e_ob]) * (rollout_eps // len(mine_locs[*e_ob])))

#                     # print('\n\n\n')

#             # print('\n\n')
#             # print(f'Rollout Q(s, a): {Q_table[*e_ob]}')
#             # print(f'Agent Q(s, a): {self.agent.Q_table[*e_ob]}')
#             # print('\n\n')

#         # raise ValueError('Rollouts not implemented yet.')

#         # Compute characteristic values
#         self.exact = defaultdict(dict)

#         # Loop over all coalitions and states
#         for C in tqdm(self.all_C(), f'Exact {self.__class__.__name__}'):
#             for e_ob in e_obs:

#                 # sum [ pi_C * Q(s, a) ]
#                 self.exact[*C][*e_ob] = (self.exact_pi(e_ob, e_ob, C) * Q_table[*e_ob]).sum(keepdims=True)
                
#         if save:
#             with open(f'{self.__class__.__name__}_char.pkl', "wb") as f:
#                 pickle.dump(self.exact, f)

#         if disp:
#             print(f'Exact characteristic values: {self.exact}')