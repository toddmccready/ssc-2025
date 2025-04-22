from typing import Dict

import jax.numpy as jnp
import jax.random as jr
import polars as pl
from bayinx.core import optimize_model
from jaxtyping import Array, Key
from numpy.lib.npyio import NpzFile
from polars import DataFrame

from scripts.analysis.model import CensoredModel, get_intervals

# Read in data
file: NpzFile = jnp.load("data/array_data.npz")  # pyright: ignore

# Extract data
predictors: Array = jnp.array(file["arr_0"]).T
outcome: Array = jnp.array(file["arr_1"])
censored: Array = jnp.array(file["arr_2"])
censoring_time: Array = jnp.array(file['arr_3'])

# Partition data into training and testing sets
key: Key = jr.PRNGKey(0)

idx: Array = jr.permutation(key, predictors.shape[1])
n_train: int = int(predictors.shape[1] * 0.8)
n_test: int = predictors.shape[1] - n_train

predictors_train, predictors_test = (
    predictors[:, idx[:n_train]],
    predictors[:, idx[-n_test:]],
)
censored_train, censored_test = censored[idx[:n_train]], censored[idx[-n_test:]]
outcome_train, outcome_test = outcome[idx[:n_train]], outcome[idx[-n_test:]]
censoring_time_train, censoring_time_test = censoring_time[idx[:n_train]], censoring_time[idx[-n_test:]]

# Format data into dictionary
data_train: Dict[str, Array] = {"predictors": predictors_train, "outcome": outcome_train, "censoring_time": censoring_time_train}
data_test: Dict[str, Array] = {"predictors": predictors_test, "outcome": outcome_test, "censoring_time": censoring_time_test}

# Construct model and optimize model
model: CensoredModel = CensoredModel()
model = optimize_model(model, int(5e4), data_train, 1e-4)
# model.eval(data_train)

# Compute training set coverage frequencies on uncensored individuals
alpha: Array = jnp.linspace(0.5, 0.01, 13)
coverage_train: Array = get_intervals(
    model,
    {
        "predictors": data_train["predictors"][:,~censored_train],
        "alpha": alpha
    },
)
coverage_train: Array = (
    (coverage_train[:, 0, :] < data_train["outcome"][~censored_train])
    & (data_train["outcome"][~censored_train] < coverage_train[:, 1, :])
).mean(1)

# Compute testing set coverage frequencies on uncensored individuals
coverage_test: Array = get_intervals(
    model,
    {"predictors": data_test["predictors"][:,~censored_test], "alpha": alpha},
)
coverage_test: Array = (
    (coverage_test[:, 0, :] < data_test["outcome"][~censored_test])
    & (data_test["outcome"][~censored_test] < coverage_test[:, 1, :])
).mean(1)

# Format results
results: DataFrame = pl.DataFrame({
    'target': (1.0 - alpha).tolist(),
    'training_coverage': coverage_train.tolist(),
    'testing_coverage': coverage_test.tolist()
})
