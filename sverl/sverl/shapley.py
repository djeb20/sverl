from math import comb
import numpy as np
from tqdm import tqdm
import pickle

class Shapley:
    def __init__(self, char):
        self.char = char

    def normalise(self, obs, sv):

        # Normalisation fix for kernel-style Shapley value calculation 
        norm_fix = (self.char.get_exact_val(obs, np.ones_like(obs)) 
                    - self.char.null(obs) 
                    - sv.sum(axis=-2)) / self.char.F_card

        return sv + np.expand_dims(norm_fix, axis=-2)
    
    def get_exact(self, disp=False, e_obs=None):
        """
        Returns the exact Shapley values using weighted least squares (WLS, similar to KernelSHAP but exact).
        """

        # Coalition matrix (m x F_card)
        X = self.char.all_C()

        # Calculate diagonal matrix of weights for each coalition based on its size. (m x m)
        weight = lambda C: (self.char.F_card - 1) / (comb(self.char.F_card, C.sum()) * C.sum() * (self.char.F_card - C.sum()))
        W = np.diag([weight(C) if (C.sum() != 0 and C.sum() != self.char.F_card) else 0 for C in X.astype(int)])

        # Compute for all states unless given.
        if e_obs is None:
            e_obs = np.unique(self.char.steady_state, axis=0)

        self.exact = {}
        for e_ob in tqdm(e_obs, 'Exact Shapley values'):

            # Characteristic values for each coalition: Y (n x m)
            Y = np.stack([self.char.get_exact_val(e_ob, C) for C in X]).T

            # Solve WLS for each output dimension of the characteristic function (F_card x n, e.g. n_actions for policy)
            solutions = np.stack([np.linalg.solve(X.T @ W @ X, X.T @ W @ y) for y in Y]).T
            self.exact[*e_ob] = self.normalise(e_ob, solutions)

        with open(f'{self.__class__.__name__.lower()}_shapley.pkl', "wb") as f:
            pickle.dump(self.exact, f)

        if disp:
            print(f'Exact Shapley Values: {self.exact}')

# Each Shapley has their own class to avoid confusion and ensure consistent naming.

class PolicyShapley(Shapley):
    def __init__(self, char):
        super().__init__(char)

class ValueShapley(Shapley):
    def __init__(self, char):
        super().__init__(char)

class PerformanceShapley(Shapley):
    def __init__(self, char):
        super().__init__(char)