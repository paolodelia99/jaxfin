import jax.numpy as jnp


class GeometricBrownianMotion:

    def __init__(self, s0, mean, vols, dtype):
        self.s0 = s0
        self.mean = mean
        self.vols = vols
        self.dtype = dtype

    def simulate_bs(self, maturity, n, n_sim):
        #TODO: add check dtype

        # delta time
        dt = maturity / n

        # initialize
        x = jnp.zeros((n_sim, n + 1))  # X = rt + X(t)

        # sample standard normal
        z = jnp.random.randn(n_sim, n)  # sample all random variables

        for i in range(n):
            x[:, i + 1] = x[:, i] + (self.mean - self.vols ** 2 / 2) * dt + self.vols * jnp.sqrt(dt) * z[:, i]

        # from logreturns to prices: X -> S
        return self.s0 * jnp.exp(x)


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
