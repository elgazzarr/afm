import numpy as np
import jax
import jax.random as jrandom
import jax.numpy as jnp
import flax.nnx as nnx
from models.flow import Flow
from trainers.flow_eval import FlowEvaluator
from gluonts.model.predictor import Predictor, OutputTransform
from gluonts.transform import SelectFields, Transformation
from typing import Iterator, List, Optional, Union, Any
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.transform import  InstanceSplitter, TestSplitSampler
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    InstanceSplitter,
    TestSplitSampler,
    SelectFields,
)
from gluonts.evaluation import MultivariateEvaluator
from data.gluonts_loaders import batchify, _to_dataframe

class JaxPredictor(Predictor):
    def __init__(
        self,
        config,
        flow: nnx.Module,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        input_names: List[str] = ["past_target"],
        required_fields: List[str] = ["forecast_start", "past_time_feat", "future_time_feat"],
        scaler: Any = None,
        rngs: Any = None,
    ) -> None:
        self.config = config
        self.input_names = input_names
        self.batch_size = batch_size
        self.scaler = scaler
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.flow = flow
        self.required_fields = required_fields
        self.rngs = rngs    

    @property
    def network(self):
        return self.prediction_net
    
    def predict(self, dataset: Dataset) -> Iterator[Forecast]:

        test_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_TIME],
        ) 

        inference_data_loader = InferenceDataLoader(dataset=dataset.input,
                                        transform=test_splitter + SelectFields(self.input_names + self.required_fields, allow_missing=True),
                                        batch_size=self.batch_size,
                                        stack_fn=batchify)
    

        for batch in inference_data_loader:
            history = jnp.array(batch["past_target"])
            future_covariates = jnp.array(batch["future_time_feat"])
            past_covariates = jnp.array(batch["past_time_feat"])
            covariates = jnp.concatenate([past_covariates, future_covariates], axis=1)
            start_date = batch["forecast_start"]
            predictions = self.flow.sample_flow(history, covariates, self.prediction_length, self.config.nfe, self.config.mcmc_samples, self.rngs())
            predictions = jnp.swapaxes(predictions, 0, 1)
            predictions_scaled = jax.vmap(jax.vmap(self.scaler.inverse_transform))(predictions)
            predictions_np = np.array(predictions_scaled)
            for i in range(predictions_np.shape[0]):
                yield SampleForecast(
                    samples = predictions_np[i],
                    start_date=start_date[i],
                    item_id=None,
                    info={}
                )
            


    def evaluate(self, targets_iter, forecasts_iter, num_series):
            targets = list(targets_iter)
            forecasts = list(forecasts_iter)
            evaluator_ = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], 
                                    target_agg_funcs={'sum': np.sum})
            agg_metric, item_metrics = evaluator_(targets, forecasts, num_series=num_series)
            return agg_metric, item_metrics

