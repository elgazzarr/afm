import jax
import jax.numpy as jnp
from jax import random, vmap


def generate_brownian_timeseries(key, N, T, scale=1.0):
    """
    Generate N Brownian motion time series of length T
    scale: scale of the Brownian motion
    """
    key_noise = random.split(key, N)

    def single_brownian(noise_key):
        # Generate one Brownian motion series
        noise = random.normal(noise_key, (T,))
        series = jnp.cumsum(noise, axis=0)
        return series

    # Generate N series in parallel
    series = vmap(single_brownian)(key_noise)
    return series  # Shape (N, T)


def generate_ar_timeseries(key, N, T, phi=0.8, sigma=1.0):
    """
    Generate N AR(1) time series of length T
    phi: autocorrelation parameter
    sigma: noise scale
    """
    key_noise = random.split(key, N)

    def single_ar(noise_key):
        # Generate one AR(1) series
        innovations = random.normal(noise_key, (T,)) * sigma
        series = jnp.zeros(T)

        def step(state, t):
            return phi * state + innovations[t], phi * state + innovations[t]

        _, series = jax.lax.scan(step, 0., jnp.arange(T))
        return series

    # Generate N series in parallel
    series = vmap(single_ar)(key_noise)
    return series  # Shape (N, T)

def generate_poisson_timeseries(key, N, T, rate=1.0):
    """
    Generate N Poisson process time series of length T
    rate: average event rate
    """
    return random.poisson(key, shape=(N, T), lam=rate)

def generate_exp_growth_timeseries(key, N, T, growth_rate=0.1, noise_scale=0.1):
    """
    Generate N exponential growth time series of length T with noise
    growth_rate: base growth rate
    noise_scale: scale of multiplicative noise
    """
    time = jnp.arange(T)
    base_curve = jnp.exp(growth_rate * time)
    noise = jnp.exp(random.normal(key, (N, T)) * noise_scale)
    return base_curve[None, :] * noise

def generate_seasonal_timeseries(key, N, T, period=24, amplitude=1.0, trend=0.1, noise_scale=0.1):
    """
    Generate N seasonal time series of length T
    period: length of seasonal cycle
    amplitude: size of seasonal fluctuations
    trend: linear trend component
    noise_scale: scale of additive noise
    """
    time = jnp.arange(T)
    seasonal = amplitude * jnp.sin(2 * jnp.pi * time / period)
    trend_component = trend * time
    base_curve = seasonal + trend_component
    noise = random.normal(key, (N, T)) * noise_scale
    return base_curve[None, :] + noise
