# Exotic Pricing library in Python

[Library name] is a powerful and versatile Python library designed for pricing exotic options using a range of advanced financial techniques.

## Quickstart

### How to get started with the development

Make sure that you've created a virtual environment with the following command

        python -m venv venv

and then install the packages needed 

       pip install -r requirements.txt

# Road map

Since using plain python is for kids we are going to use (or least try to use) on of the two following libraries
to make our computation performant (even if we are using python):

- Jax
  - the problem with jax is that it doesn't support the NVIDIA GPU on windows 
- CuPy
  - probably this is going to make our code more verbose

## What we would like to support

Providing pricing capabilities for the following types of exotic options:

1. **European Options:** Standard options that can be exercised only at expiration.

2. **American Options:** Options that can be exercised at any time before or at expiration.

3. **Asian Options:** Options whose payoff depends on the average price of the underlying asset over a specified period.

4. **Lookback Options:** Options whose payoff is based on the extrema (maximum or minimum) of the underlying asset's price over its life.

5. **Down and Out Options:** Options that expire worthless if the underlying asset's price falls below a predetermined barrier during the option's life.

6. **Knock and Out Options:** Options that expire worthless if the underlying asset's price reaches a specified barrier during the option's life.

7. **Up and Out Options:** Options that expire worthless if the underlying asset's price rises above a predetermined barrier during the option's life.

## Pricing Techniques

### 1. Monte Carlo Simulation

Utilizes random sampling to simulate multiple possible paths of the underlying asset's price and calculates the option's expected payoff.

### 2. Partial-Integro Differential Equations

Involves solving partial-integro differential equations to model the behavior of exotic options over time.

### 3. Carr-Madan Algorithm

An efficient numerical algorithm used for option pricing, particularly in the context of Fourier-based methods.

### 4. Convolution Method

Leverages convolution techniques to compute the option's price efficiently.

# TODOs

- [ ] Separate BS and Black model in two different files
- [ ] Add tests 
- [ ] see if it possible to use some fwd_propagation to calculate the greeks of the european
- [ ] see where it is possible to jit the `simulate_paths` of UnivariateGBM
- [ ] use tox for automation stuff
- [ ] create the package