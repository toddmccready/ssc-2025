import jax.numpy as jnp
import jax.random as jr
import polars as pl
from bayinx.core import optimize_model
from jaxtyping import Array, Key
from numpy.lib.npyio import NpzFile
from polars import DataFrame

from scripts.analysis.model import UncensoredModel, get_intervals

# Read in data
file: NpzFile = jnp.load("data/array_data.npz")  # pyright: ignore

# Extract data
predictors = jnp.array(file["arr_0"])
outcome = jnp.array(file["arr_1"])
censored = jnp.array(file["arr_2"])

# Subset uncensored individuals
predictors = predictors[~censored].T
outcome = outcome[~censored]

# Partition data into training and testing sets
key: Key = jr.PRNGKey(0)
idx = jr.permutation(key, predictors.shape[1])
n_train = int(predictors.shape[1] * 0.8)
n_test = predictors.shape[1] - n_train

predictors_train, predictors_test = (
    predictors[:, idx[:n_train]],
    predictors[:, idx[-n_test:]],
)
outcome_train, outcome_test = outcome[idx[:n_train]], outcome[idx[-n_test:]]

# Format data into dictionary
data_train = {"predictors": predictors_train, "outcome": outcome_train}
data_test = {"predictors": predictors_test, "outcome": outcome_test}

# Construct model and optimize model
model: UncensoredModel = UncensoredModel()
model = optimize_model(model, int(1e3), data_train, 1e-3)
# model.eval(data_train)

# Compute training set coverage frequencies
alpha: Array = jnp.linspace(0.5, 0.01, 13)
coverage_train: Array = get_intervals(
    model,
    {
        "predictors": data_train["predictors"],
        "alpha": alpha
    },
)
coverage_train: Array = (
    (coverage_train[:, 0, :] < data_train["outcome"])
    & (data_train["outcome"] < coverage_train[:, 1, :])
).mean(1)

# Compute testing set coverage frequencies
coverage_test: Array = get_intervals(
    model,
    {"predictors": data_test["predictors"], "alpha": alpha},
)
coverage_test: Array = (
    (coverage_test[:, 0, :] < data_test["outcome"])
    & (data_test["outcome"] < coverage_test[:, 1, :])
).mean(1)


results: DataFrame = pl.DataFrame({
    'alpha': alpha.tolist(),
    'training_coverage': coverage_train.tolist(),
    'testing_coverage': coverage_test.tolist()
})
