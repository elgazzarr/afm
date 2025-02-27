import gluonts
from gluonts.dataset.repository import get_dataset, dataset_recipes
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
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
from typing import List, Optional, Callable, Iterable, Tuple, Iterator, Any
import jax.numpy as jnp
from functools import partial
import numpy as np
import pandas as pd
from data.dataloader import GluonTSDataLoaderAdapter
from gluonts.transform import  SimpleTransformation
import numpy as np
from gluonts.dataset import DataBatch, Dataset
from data.scalers import StandardScaler, NothingScaler

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

def create_date_embeddings(start_date_str, n_timepoints, freq):
    # Convert string to datetime if needed
    if isinstance(start_date_str, str):
        start_date = pd.to_datetime(start_date_str)
    else:
        start_date = start_date_str

    # Convert Period to Timestamp (Fix for the ValueError)
    if isinstance(start_date, pd.Period):
        start_date = start_date.to_timestamp()

    # Generate timestamps based on frequency
    if freq == "H":
        timestamps = pd.date_range(start=start_date, periods=n_timepoints, freq="H")
    elif freq == "D":
        timestamps = pd.date_range(start=start_date, periods=n_timepoints, freq="D")
    elif freq == "B":
        timestamps = pd.date_range(start=start_date, periods=n_timepoints, freq="B")
    else:
        raise ValueError("Unsupported frequency. Use 'H', 'D', or 'B'.")

    # Initialize arrays for each feature
    day_sin = np.zeros(n_timepoints)
    day_cos = np.zeros(n_timepoints)
    month_sin = np.zeros(n_timepoints)
    month_cos = np.zeros(n_timepoints)
    year_sin = np.zeros(n_timepoints)
    year_cos = np.zeros(n_timepoints)

    if freq == "H":
        hour_sin = np.zeros(n_timepoints)
        hour_cos = np.zeros(n_timepoints)
    
    # Fill arrays with embeddings for each timestamp
    for i, ts in enumerate(timestamps):
        if freq == "H":
            hour_sin[i] = np.sin(2 * np.pi * ts.hour / 24.0)
            hour_cos[i] = np.cos(2 * np.pi * ts.hour / 24.0)

        day_sin[i] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
        day_cos[i] = np.cos(2 * np.pi * ts.dayofweek / 7.0)
        month_sin[i] = np.sin(2 * np.pi * (ts.month - 1) / 12.0)
        month_cos[i] = np.cos(2 * np.pi * (ts.month - 1) / 12.0)
        year_sin[i] = np.sin(2 * np.pi * (ts.dayofyear - 1) / 365.0)
        year_cos[i] = np.cos(2 * np.pi * (ts.dayofyear - 1) / 365.0)

    # Stack all features (include hour features only for hourly frequency)
    if freq == "H":
        return np.stack([hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, year_sin, year_cos])
    else:
        return np.stack([day_sin, day_cos, month_sin, month_cos, year_sin, year_cos])

def add_temporal_covariates(dataset: Iterable[dict], freq) -> Iterable[dict]:
    for item in dataset:
        n_timepoints = item['target'].shape[1]
        date_covariates = create_date_embeddings(item['start'], n_timepoints, freq=freq)
        item['time_feat'] = date_covariates

    return dataset


def get_dataset_loader(dataset_name: str, batch_size: int, context_ratio: int, scale=False) -> Tuple[GluonTSDataLoaderAdapter, GluonTSDataLoaderAdapter, Any, dict, SimpleTransformation]:

    dataset = get_dataset(dataset_name, regenerate=False)
    dataset_freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    context_length = prediction_length * context_ratio

    train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                    max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
    
    dataset_train = add_temporal_covariates(train_grouper(dataset.train), dataset_freq)
    dataset_test = add_temporal_covariates(test_grouper(dataset.test), dataset_freq)

    TOTAL_LEN = dataset_train[0]['target'].shape[1]
    VAL_START = TOTAL_LEN - prediction_length - context_length
    VAL_SIZE = TOTAL_LEN - VAL_START

    scaler = StandardScaler(field_name=FieldName.TARGET) if scale else NothingScaler()
    scaler.fit(dataset_train)
    dataset_train = [scaler.transform(entry.copy()) for entry in dataset_train]
    
    train_transform = Chain([
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

    train_data_loader = GluonTSDataLoaderAdapter(
        TrainDataLoader(
            dataset=Cached(dataset_train),
            batch_size=batch_size,
            transform=train_transform,
            stack_fn=batchify
        )
    )

    val_data_loader = GluonTSDataLoaderAdapter(
        InferenceDataLoader(
            dataset=Cached(dataset_train),
            batch_size=1,
            transform=val_transform,
            stack_fn=batchify
        )
    )


    # Preparing test_loader

    window_length = prediction_length 
    _, test_template = split(dataset_test, offset=-window_length)
    test_data = test_template.generate_instances(window_length)


    gt_it = map(_to_dataframe, test_data),

    context, covariates, _ = next(iter(train_data_loader))
    
    dims = {"data_dim": context.shape[-1], "cov_dim": covariates.shape[-1]}
    

    return train_data_loader, val_data_loader, test_data, dims, scaler