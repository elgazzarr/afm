import diffrax as dfx
from seq_priors import generate_brownian_timeseries
import jax.random as jrandom
import jax.numpy as jnp
from models import NoARFlow, ARFlow
import jax

def sample_flow(model, key, history, covariates, n_steps_flow, n_steps_pred, n_samples):

    def predict_noar(key, history, covariates, n_steps_flow, n_steps_pred):
        def vf(t, x, args):
            history, covariates = args
            return model(x, history, covariates, t)
            
        term = dfx.ODETerm(vf)
        solver = dfx.Euler()
        dt = 1/n_steps_flow
        x0 = generate_brownian_timeseries(key, history.shape[-1], n_steps_pred).T
        predictions = dfx.diffeqsolve(term, solver, 0., 1., dt, x0, args=(history, covariates), saveat=dfx.SaveAt(t1=True)).ys
        return predictions
    
    def predict_ar(key, history, covariates, n_steps_flow, n_steps_pred):
        def vf(t, x, args):
            history, covariates = args
            return model(x, history, covariates, jnp.array([t]))
            
        term = dfx.ODETerm(vf)
        solver = dfx.Euler()
        dt = 1/n_steps_flow
        predictions = []
        current_history = history
        context_length = history.shape[0]
        for _ in range(n_steps_pred):
            context = current_history[-context_length:]
            x0 = jrandom.normal(key, shape=context.shape[-1])
            x_next = dfx.diffeqsolve(
                term, solver, 0., 1., dt, x0, 
                args=(context, covariates), 
                saveat=dfx.SaveAt(t1=True)
            ).ys
            predictions.append(x_next)
            current_history = jnp.concatenate([current_history, x_next], axis=0)
            
        predictions = jnp.vstack(predictions)
        return predictions

    key_samples = jrandom.split(key, n_samples)

    if isinstance(model, NoARFlow):
        predict = predict_noar
    else:
        predict = predict_ar
    
    # First vmap over batch, then over samples
    vmapped_predict = jax.vmap(predict, in_axes=(None, 0, 0, None, None))  # vmap over batch
    vmapped_predict = jax.vmap(vmapped_predict, in_axes=(0, None, None, None, None))  # vmap over samples
    
    pred_samples = vmapped_predict(key_samples, history, covariates, n_steps_flow, n_steps_pred)
    return pred_samples


