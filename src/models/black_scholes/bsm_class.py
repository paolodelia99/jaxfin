import numpy as np


class BlackScholesSimulator:

    """
    Simulation of the underlying according to B&S model, i.e. following a Geometric Brownian Motion
    -----
    S0:     price at time 0 [float]
    T:      maturity in yearfrac [float]
    r:      risk free rate [float]
    N_SIM:  number of simulation [int]
    N:      number of the monitoring data [int]
    SIGMA:  underlying volatility [float]
    """

    def __init__(self, S0, T, r, N_SIM, N, SIGMA):
        self.S0 = S0
        self.T = T
        self.r = r
        self.N_SIM = N_SIM
        self.N = N
        self.SIGMA = SIGMA

    def simulate_bs(self):
        # delta time
        dt = self.T / self.N

        # initialize
        x = np.zeros((self.N_SIM, self.N + 1))  # X = rt + X(t)

        # sample standard normal
        z = np.random.randn(self.N_SIM, self.N)  # sample all random variables

        for i in range(self.N):
            x[:, i + 1] = x[:, i] + (self.r - self.SIGMA ** 2 / 2) * dt + self.SIGMA * np.sqrt(dt) * z[:, i]

        # from logreturns to prices: X -> S
        return self.S0 * np.exp(x)


"""
        # Example usage:
        params = {
            'S0': 100,
            'T': 1,
            'r': 0.05,
            'N_SIM': 1000,
            'N': 252,
            'SIGMA': 0.2
        }
        
        bs_simulator = BlackScholesSimulator(**params)
        result = bs_simulator.simulate_bs()
        print(result)
"""
