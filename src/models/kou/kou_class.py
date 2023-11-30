import numpy as np
from scipy.stats import poisson, expon

class KouSimulator:
    def __init__(self, SIGMA, LAMBDA, LAMBDA_MINUS, LAMBDA_PLUS, P):
        self.SIGMA = SIGMA
        self.LAMBDA = LAMBDA
        self.LAMBDA_MINUS = LAMBDA_MINUS
        self.LAMBDA_PLUS = LAMBDA_PLUS
        self.P = P

    def charexp(self, u):
        return -self.SIGMA**2 * u**2 / 2 + 1j * u * self.LAMBDA * (
            self.P / (self.LAMBDA_PLUS - 1j * u) - (1 - self.P) / (self.LAMBDA_MINUS + 1j * u)
        )

    def simulate_kou(self, S0, T, r, N_SIM, N):
        # delta time
        dt = T / N

        # initialize X
        X = np.zeros((N_SIM, N + 1))  # X = rt + X(t)

        # number of jumps
        NT = poisson.ppf(np.random.rand(N_SIM, 1), self.LAMBDA * T)

        # characteristic exponent
        drift = r - self.charexp(-1j)  # in order to be risk-neutral

        for j in range(N_SIM):
            JumpTimes = np.sort(T * np.random.rand(int(NT[j]), 1))

            for i in range(N):
                # add diffusion component
                X[j, i + 1] = X[j, i] + drift * dt + self.SIGMA * np.sqrt(dt) * np.random.randn()

                # add jump part -> ( (i-1)dt, idt )
                for l in range(int(NT[j])):
                    if (i - 1) * dt < JumpTimes[l] <= i * dt:
                        sim_p = np.random.rand()

                        if sim_p < self.P:  # positive jump
                            Y = expon.ppf(np.random.rand(), scale=1 / self.LAMBDA_PLUS)
                        else:  # negative jump
                            Y = -expon.ppf(np.random.rand(), scale=1 / self.LAMBDA_MINUS)

                        X[j, i + 1] = X[j, i + 1] + Y

        S = S0 * np.exp(X)

        return S

"""
# Example usage:
kou_simulator = KouSimulator(0.2, 0.1, 0.05, 0.15, 0.8)  # Replace with the actual values
result = kou_simulator.simulate_kou(100, 1, 0.05, 1000, 252)
print(result)
"""