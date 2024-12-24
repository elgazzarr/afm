import gluonts
from gluonts.dataset.repository import get_dataset, dataset_recipes
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddObservedValuesIndicator, InstanceSplitter, ExpectedNumInstanceSampler, TestSplitSampler, ValidationSplitSampler
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    Transformation,
    Chain,
    InstanceSampler,
    SelectFields,
)
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from typing import List, Optional, Callable, Iterable, Tuple, Iterator
import jax.numpy as jnp
from functools import partial
import numpy as np
import pandas as pd
from data.dataloader import GluonTSDataLoaderAdapter
from gluonts.transform import  SimpleTransformation
import numpy as np

class StandardScaler(SimpleTransformation):
    def __init__(
        self,
        dim=0,  # dimension 0 for channels in [channels, time]
        keepdim=True,
        minimum_scale=1e-10,
        field_name="target",
    ):
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.field_name = field_name
        self.mean = None
        self.std = None

    def fit(self, data_iterator):
        # First pass: compute means per channel
        sum_x = None
        count = 0
        
        for data in data_iterator:
            target = data[self.field_name]  # Shape: [channels, time]
            # Sum along time dimension (1) to get per-channel statistics
            if sum_x is None:
                sum_x = np.sum(target, axis=1, keepdims=self.keepdim)  # [channels, 1]
            else:
                sum_x += np.sum(target, axis=1, keepdims=self.keepdim)
            count += target.shape[1]  # Add number of timesteps
        
        self.mean = sum_x / count  # [channels, 1]
        
        # Second pass: compute std per channel
        sum_squared_dev = None
        for data in data_iterator:
            target = data[self.field_name]
            squared_dev = (target - self.mean) ** 2
            if sum_squared_dev is None:
                sum_squared_dev = np.sum(squared_dev, axis=1, keepdims=self.keepdim)
            else:
                sum_squared_dev += np.sum(squared_dev, axis=1, keepdims=self.keepdim)
        
        self.std = np.sqrt(sum_squared_dev / count)
        self.std = np.maximum(self.std, self.minimum_scale)

    def transform(self, data):
        target = data[self.field_name]  # [channels, time]
        scaled_data = (target - self.mean) / self.std
        data[self.field_name] = scaled_data
        data['scale'] = self.std  # [channels, 1]
        data['loc'] = self.mean   # [channels, 1]
        return data
    
    def transform_instance(self, data: jnp.ndarray) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")            
        data = jnp.transpose(data)
        scaled_data = (data - self.mean) / self.std
        return jnp.transpose(scaled_data)
    
    def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse transform")            
        data = jnp.transpose(data)
        original_scale_data = (data * self.std) + self.mean
        return jnp.transpose(original_scale_data)


def _to_dataframe(input_label: Tuple[DataEntry, DataEntry]) -> pd.DataFrame:
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def stack(data, device=None):
    if isinstance(data[0], (np.ndarray, jnp.ndarray)):
        return np.array(data)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t) for t in zip(*data))
    return data

def batchify(data: List[dict]) -> dict:
    return {
        key: stack(data=[item[key] for item in data])
        for key in data[0].keys()
    }

def create_date_embeddings(start_date_str, n_timepoints):
    # Convert string to datetime if needed
    if isinstance(start_date_str, str):
        start_date = pd.to_datetime(start_date_str)
    else:
        start_date = start_date_str
    
    # Create array of sequential timestamps - using hours now
    timestamps = [start_date + pd.Timedelta(hours=i) 
                 for i in range(n_timepoints)]
    
    # Initialize arrays for each feature
    hour_sin = np.zeros(n_timepoints)
    hour_cos = np.zeros(n_timepoints)
    day_sin = np.zeros(n_timepoints)
    day_cos = np.zeros(n_timepoints)
    month_sin = np.zeros(n_timepoints)
    month_cos = np.zeros(n_timepoints)
    
    # Fill arrays with embeddings for each timestamp
    for i, ts in enumerate(timestamps):
        hour_sin[i] = np.sin(2 * np.pi * ts.hour / 24.0)
        hour_cos[i] = np.cos(2 * np.pi * ts.hour / 24.0)
        day_sin[i] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
        day_cos[i] = np.cos(2 * np.pi * ts.dayofweek / 7.0)
        month_sin[i] = np.sin(2 * np.pi * (ts.month - 1) / 12.0)
        month_cos[i] = np.cos(2 * np.pi * (ts.month - 1) / 12.0)
    
    # Stack all features
    return np.stack([hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos])


def add_temporal_covariates(dataset: Iterable[dict]) -> Iterable[dict]:
    for item in dataset:
        n_timepoints = item['target'].shape[1]
        date_covariates = create_date_embeddings(item['start'], n_timepoints)
        item['time_feat'] = date_covariates
    return dataset


def get_dataset_loader(dataset_name: str, batch_size: int, context_ratio: int) -> Tuple[GluonTSDataLoaderAdapter, GluonTSDataLoaderAdapter, InferenceDataLoader, dict, SimpleTransformation]:

    dataset = get_dataset(dataset_name, regenerate=False)
    prediction_length = dataset.metadata.prediction_length
    context_length = prediction_length * context_ratio
    freq = dataset.metadata.freq
    input_names = ["past_target"]
    required_fields = ["forecast_start", "past_time_feat", "future_time_feat"]

    train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                    max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
    
    dataset_train = add_temporal_covariates(train_grouper(dataset.train))
    dataset_test = add_temporal_covariates(test_grouper(dataset.test))

    TOTAL_LEN = dataset_train[0]['target'].shape[1]
    VAL_START = TOTAL_LEN - prediction_length - context_length
    VAL_SIZE = TOTAL_LEN - VAL_START

    scaler = StandardScaler(field_name=FieldName.TARGET)
    scaler.fit(dataset_train)
    
    train_transform = Chain([
        scaler
        ,
        AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,)
        ,
        InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_future=prediction_length, max_future=VAL_START),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_TIME])
        ])
    
    val_transform = Chain([
        scaler
        ,
        AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,)
        ,
        InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ValidationSplitSampler(min_past=VAL_START, min_future=prediction_length),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_TIME])
        ])
    
    test_transform = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.FEAT_TIME],
    )  

    train_data_loader = GluonTSDataLoaderAdapter(TrainDataLoader(
        dataset = Cached(dataset_train),
        batch_size = batch_size,
        transform = train_transform,
        stack_fn = batchify)
    )

    val_data_loader = GluonTSDataLoaderAdapter(InferenceDataLoader(
        dataset = Cached(dataset_train),
        batch_size = 1,
        transform = val_transform,
        stack_fn = batchify
    ))

    test_data_loader = InferenceDataLoader(
        dataset = Cached(dataset_test),
        batch_size = batch_size,
        transform = test_transform +  SelectFields(input_names + required_fields, allow_missing=True),
        stack_fn = batchify
    )

    context, covariates, target = next(iter(train_data_loader))

    dims = {"data_dim": context.shape[-1], "cov_dim": covariates.shape[-1]}
    
    return train_data_loader, val_data_loader, test_data_loader, dims, scaler