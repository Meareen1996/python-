from math import sqrt, log, exp
from scipy.stats import norm
import numpy as np

'''
The price_path method generates a Monte Carlo price path, which is then used in the payoff method to calculate the payoff. 
If at any point in time S(ti) is greater than U, the option terminates and the option holder is paid R immediately. 
Otherwise, if at some point in time S(ti) is less than or equal to L, max(K - S(T), 0) is paid at expiration, where S(T) is the average of the asset prices at expiration.
'''

class KIKOPut:
    def __init__(self, S, K, L, U, R, T, n, sigma, r, M):
        self.S = S
        self.K = K
        self.L = L
        self.U = U
        self.R = R
        self.T = T
        self.n = n
        self.sigma = sigma
        self.r = r
        self.M = M

    def price_path(self):
        Dt = self.T / self.n
        np.random.seed(100)
        Z = np.random.randn(self.M, self.n)
        drift = (self.r - 0.5 * self.sigma**2) * Dt
        vol = self.sigma * np.sqrt(Dt)
        price_path = self.S * np.cumprod(np.exp(drift + (vol * Z)), 1)
        return price_path

    def payoff(self):
        sPath = self.price_path()
        for i in range(1, self.n + 1):
            ti = i / self.n * self.T
            if sPath[:, i-1].max() >= self.U:
                return self.R
            elif sPath[:, i-1].min() <= self.L:
                return max(self.K - sPath[:, -1].mean(), 0)
        return 0