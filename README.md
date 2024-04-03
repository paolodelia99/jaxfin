# JaxFin

`JaxFin` is a powerful and versatile Python library designed for pricing exotic options using a range of advanced financial techniques. The library is built with the aim of providing a robust, flexible, and efficient tool for quantitative finance professionals, researchers, and students alike. It offers a comprehensive suite of features that allow users to model, price, and analyze a wide variety of exotic options.

The core strength of `JaxFin` lies in its use of the [jax](https://github.com/google/jax) library. Jax is a high-performance library for accelerated array computing, offering features such as automatic differentiation (AutoGrad), accelerated linear algebra (XLA), and just-in-time compilation to GPU/TPU. By leveraging these capabilities, `JaxFin` is able to perform complex financial computations with exceptional speed and efficiency.

Whether you're looking to price a complex exotic option, perform a large-scale Monte Carlo simulation, or simply explore advanced financial models, `JaxFin` provides the tools and performance you need.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Building the Library Locally](#building-the-library-locally)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install the project using pip:

```bash
pip install JaxFin
```

## Quickstart

You now might wonder why use `jaxfin` in first place, what's the advantage to any other quantitative finance library available in Python? Well if don't know it `jax` is basically `NumPy`, on steroids, which in short that  means it provides the familiar and intuitive interface of NumPy but with the added benefits of automatic differentiation, GPU/TPU support, and JIT compilation. 

`jaxfin` combines the best of both worlds: the power and performance of `jax`, with the simplicity and familiarity of standard Python libraries. This makes it a powerful tool for any quantitative finance professional or enthusiast.

Here's a small example to illustrate of you can use `jaxfin` to price a European option using the Black-Scholes model:

```python
import jax.numpy as jnp
from jaxfin.price_engine.black_scholes import european_price

# Black-Scholes price of an european option
spots = jnp.asarray([100])
strikes = jnp.asarray([110])
expires = jnp.asarray([1.0])
vols = jnp.asarray([0.2])
discount_rates = jnp.asarray([0.0])
european_price(spots, strikes, expires, vols, discount_rates, dtype=jnp.float32)
```

Computing the price price of a single option, nahh that's boring! what we can do with jaxfin is to compute the price of a basket of options leveraging the vectorization capabilities of `jax` using `jax.vmap`:

```python
from jax import vmap
import jax.numpy as jnp
from jaxfin.price_engine.black_scholes import european_price

v_european_price = vmap(european_price, in_axes=(0, None, None, None, None))

# Black-Scholes price of an european option
spots = jnp.asarray([80, 90, 100, 110, 120])
strikes = jnp.asarray(110)
expires = jnp.asarray(1.0)
vols = jnp.asarray(0.2)
discount_rates = jnp.asarray(0.0)
v_european_price(spots, strikes, expires, vols, discount_rates)

>> Array([0.4424405, 1.6421748, 4.292015, 8.762122, 15.010387], dtype=float32)
```

In addition to that the calculation of the greeks, is done not throught the closed form formulas but through the automatic differentiation capabilities of `jax`. For example the function that calculates the delta of the european option under the Black Scholes model can be simply obtained as follows:

```python
delta_european = jax.grad(european_price, argnums=0)

# Black-Scholes delta of an european option
spots = jnp.asarray(100)
strikes = jnp.asarray(110)
expires = jnp.asarray(1.0)
vols = jnp.asarray(0.2)
discount_rates = jnp.asarray(0.0)
delta_european(spots, strikes, expires, vols, discount_rates, dtype=jnp.float32)
```

### Funcionalities implemented

- Price engine
    - Black scholes
        - Pricing european options
        - Greeks of european options
    - Black model
        - Pricing european options
        - Greeks of european options (just delta and gamma)
    - Fourier methods
        - Pricing european options using inverse fourier transform
        - Greeks of european options (just delta)
- Models
    - Geometric brownian motion
      - Univariate 
      - Multivariate
    - Heston
      - Univariate
      - Multivariate

## Building the Library Locally

To build the library locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/username/project-name.git
```

2. Navigate into the project directory:

```bash
cd JaxFin
```

3. Install the build dependencies:

```bash
pip install -r requirements/build.txt
```

4. Build the library:

```bash
python -m build
```

### Sanity checks

Since we want to keep the library maintainable over the long run, this implies that code has to attain to certain standards.
To achieve that the following scripts have been provided

- `scripts\run-pylint.bat`: This script runs Pylint, a tool that checks for errors in Python code, enforces a coding standard, and looks for code smells. It can also look for certain type errors, it can recommend suggestions about how particular blocks can be refactored and can offer you details about the code's complexity.
- `scripts\run-tests.bat`: This script runs the unit tests for the project.
- `scripts\run-mypy.bat`: This script runs Mypy, a static type checker for Python. Mypy can catch certain types of errors at compile time that would otherwise only be caught at runtime in standard Python. It's a way to get some of the benefits of static typing in a dynamically typed language.
- `scripts\check-black.bat`: This script runs Black, the "uncompromising" Python code formatter. By using it, you ensure that your codebase has a consistent style, which can make it easier to read and maintain.

Otherwise a `Makefile` have been provided to run all the pre-commit checks at once:

```bash
make commit-checks
```

To run the tests you can use the following command:

```bash
make pytest
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature`)
5. Create a new Pull Request

## License

This project is licensed under the [MIT License](LICENSE).
