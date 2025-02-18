from dataclasses import dataclass
from typing import Callable, Tuple
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dfx

@dataclass
class DynamicalSystem:
    vector_field: Callable
    params: jnp.ndarray
    data_dim: int
    initial_state: Callable
    solver: dfx.AbstractSolver = dfx.Tsit5()
    stochastic: bool = False

def create_system(name: str, stochastic: bool = False) -> DynamicalSystem:
    if name == 'lotka_volterra':
        def vector_field(t, x, args):
            alpha, beta, gamma, delta = args
            dx1 = alpha*x[0] - beta*x[0]*x[1]
            dx2 = -delta*x[1] + gamma*x[1]*x[0]
            return jnp.array([dx1, dx2])
        
        params = jnp.array([1.3, 0.9, 0.8, 1.8])
        data_dim = 2
        
        if stochastic:
            initial_state = lambda key, n_samples: jnp.repeat(
                jnp.array((2.0, 1.0))[None, :], n_samples, axis=0)
            solver = dfx.EulerHeun()
        else:
            initial_state = lambda key, n_samples: jrandom.uniform(
                key, (n_samples, data_dim), minval=0, maxval=5)
            solver = dfx.Tsit5()
        
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            # Environmental noise affecting both predator and prey
            diffusion = lambda t, y, args: jnp.array((0.5, 0.5))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system
    
    elif name == 'lorenz':
        def vector_field(t, x, args):
            sigma, rho, beta = args
            dx1 = sigma*(x[1] - x[0])
            dx2 = x[0]*(rho - x[2]) - x[1]
            dx3 = x[0]*x[1] - beta*x[2]
            return jnp.array([dx1, dx2, dx3])
        
        params = jnp.array([10.0, 28.0, 8/3])
        data_dim = 3
        
        if stochastic:
            initial_state = lambda key, n_samples: jnp.repeat(
                jnp.array((-6.557, -6.973, 24.059))[None, :], n_samples, axis=0)
            solver = dfx.EulerHeun()
        else:
            initial_state = lambda key, n_samples: jrandom.uniform(
                key, (n_samples, data_dim), minval=0, maxval=10)
            solver = dfx.Tsit5()
            
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            diffusion = lambda t, y, args: jnp.array((1.0, 1.0, 1.0))
            brownian_motion = dfx.VirtualBrownianTree(0, 2, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field), 
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
            
        return system
    
    elif name == 'vanderpol':
        def vector_field(t, x, args):
            mu, = args
            dx1 = x[1]
            dx2 = mu*(1 - x[0]**2)*x[1] - x[0]
            return jnp.array([dx1, dx2])
        
        params = jnp.array([0.1])
        data_dim = 2
        initial_state = lambda key, n_samples: jrandom.uniform(key, (n_samples, data_dim), minval=-2, maxval=2)
        if stochastic:
            solver = dfx.EulerHeun()
        else:
            solver = dfx.Tsit5()

        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            # Add noise primarily to the velocity component
            diffusion = lambda t, y, args: jnp.array((0.4, 0.7))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-2, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system
    
    elif name == 'harmonic_oscillator':
        def vector_field(t, x, args):
            k, m = args
            dx1 = x[1]
            dx2 = -k/m*x[0]
            return jnp.array([dx1, dx2])
        
        params = jnp.array([1.0, 1.0])
        data_dim = 2
        initial_state = lambda key, n_samples: jrandom.uniform(key, (n_samples, data_dim), minval=-2, maxval=2)

        
        if stochastic:
            solver = dfx.EulerHeun()
        else:
            solver = dfx.Tsit5()

        
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            diffusion = lambda t, y, args: jnp.array((0.5, 0.5))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-2, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system
    
    elif name == 'fitzhugh_nagumo':
        def vector_field(t, x, args):
            a, b, tau, I = args
            dx1 = x[0] - (x[0]**3) / 3 - x[1] + I
            dx2 = (x[0] + a - b * x[1]) / tau
            return jnp.array([dx1, dx2])
        
        params = jnp.array([0.7, 0.8, 12.5, 0.5])
        data_dim = 2
        initial_state = lambda key, n_samples: jrandom.uniform(
                key, (n_samples, data_dim), minval=-2, maxval=2)
        
        if stochastic:
            solver = dfx.EulerHeun()
        else:
            solver = dfx.Tsit5()
            
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            # Add noise to membrane potential and recovery variable
            diffusion = lambda t, y, args: jnp.array((0.5, 0.5))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system

    elif name == 'duffing':
        def vector_field(t, x, args):
            alpha, beta, delta, gamma, omega = args
            dx1 = x[1]
            dx2 = -delta*x[1] - alpha*x[0] - beta*x[0]**3 + gamma*jnp.cos(omega*t)
            return jnp.array([dx1, dx2])
        
        params = jnp.array([1.0, -1.0, 0.2, 0.3, 1.2])
        data_dim = 2
        initial_state = lambda key, n_samples: jrandom.uniform(
                key, (n_samples, data_dim), minval=-1, maxval=1)
        if stochastic:
            solver = dfx.EulerHeun()
        else:
            solver = dfx.Tsit5()
            
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            diffusion = lambda t, y, args: jnp.array((0.1, 0.3))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system

    elif name == 'brusselator':
        def vector_field(t, x, args):
            A, B = args
            dx1 = A + x[0]**2 * x[1] - (B + 1) * x[0]
            dx2 = B * x[0] - x[0]**2 * x[1]
            return jnp.array([dx1, dx2])
        
        params = jnp.array([1.0, 3.0])
        data_dim = 2
        initial_state = lambda key, n_samples: jrandom.uniform(key, (n_samples, data_dim), minval=0, maxval=2)
        if stochastic:
            solver = dfx.EulerHeun()
        else:
            solver = dfx.Tsit5()
            
        system = DynamicalSystem(vector_field, params, data_dim, initial_state, solver, stochastic)
        
        if stochastic:
            diffusion = lambda t, y, args: jnp.array((0.5, 0.5))
            brownian_motion = dfx.VirtualBrownianTree(0, 20, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
            system.vector_field = dfx.MultiTerm(
                dfx.ODETerm(vector_field),
                dfx.ControlTerm(diffusion, brownian_motion)
            )
        else:
            system.vector_field = dfx.ODETerm(vector_field)
        return system
    
    else:
        raise ValueError(f'System {name} not implemented')