import numpy as np
from scipy.stats import poisson


class MertonSimulator:

    """
    Simulation of the underlying according to Merton model
    -----
    S0:     price at time 0 [float]
    T:      maturity in yearfrac [float]
    r:      risk free rate [float]
    N_SIM:  number of simulation [int]
    N:      number of the monitoring data [int]
    SIGMA:  underlying volatility [float]
    PARAMS: vector of merton parameters
    """

    def __init__(self, S0, T, r, N_SIM, N, PARAMS):
        self.S0 = S0
        self.T = T
        self.r = r
        self.N_SIM = N_SIM
        self.N = N
        self.PARAMS = PARAMS

    def simulate_merton(self):
        sigma, lambda_mert, mu, delta = self.PARAMS

        # delta time
        dt = self.T / self.N

        # initialize X
        x = np.zeros((self.N_SIM, self.N + 1))  # X = rt + X(t)

        # number of jumps
        NT = poisson.ppf(np.random.rand(self.N_SIM, 1), lambda_mert * self.T)

        # characteristic exponent
        charexp = lambda u: -0.5 * sigma**2 * u**2 + lambda_mert * (
            np.exp(-0.5 * delta**2 * u**2 + 1j * mu * u) - 1
        )
        drift = self.r - charexp(-1j)  # in order to be risk-neutral

        for j in range(self.N_SIM):
            JumpTimes = np.sort(self.T * np.random.rand(int(NT[j]), 1))

            for i in range(self.N):
                # add diffusion component
                x[j, i + 1] = (
                    x[j, i] + drift * dt + sigma * np.sqrt(dt) * np.random.randn()
                )

                # add jump part if exists jump in ((i-1)*dt, i*dt)
                for l in range(int(NT[j])):
                    if (i - 1) * dt < JumpTimes[l] <= i * dt:
                        Y = mu + delta * np.random.randn()
                        x[j, i + 1] = x[j, i + 1] + Y

        S = self.S0 * np.exp(x)

        return S


"""
            # Example usage:
            params = [0.2, 0.1, 0.02, 0.1]  # Replace with the actual values
            merton_simulator = MertonSimulator(100, 1, 0.05, 1000, 252, params)
            result = merton_simulator.simulate_merton()
            print(result)
"""
