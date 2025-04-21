import jax.numpy as jnp
import polars as pl
import polars.selectors as cs

# Read in data
data = pl.read_parquet("data/ssc_competition_data.parquet")
data_dict = pl.read_parquet("data/ssc_competition_data_dict.parquet")

# Mean-imputation
vars = data_dict.filter(
    pl.col("Section category").is_in(["Laboratory", "Medications", "ECG"])
)["Variable name"]
data = data.with_columns([pl.col(name).fill_null(pl.col(name).mean()) for name in vars])

# Format missing time-to-afib
data = data.with_columns(
    censored=pl.col("outcome_afib_aflutter_new_post").eq(False),
    censoring_time=pl.col("follow_up_duration"),
    outcome=pl.when(pl.col("outcome_afib_aflutter_new_post").eq(False))
    .then(pl.col("follow_up_duration"))
    .otherwise(pl.col("time_to_outcome_afib_aflutter_new_post")),
)

# Extract predictors
vars = data_dict.filter(
    ~pl.col("Section category").is_in(["System", "Future outcomes"])
)["Variable name"]
predictors = data.select(vars.to_list())

# Convert to JAX Array
predictors = (
    predictors.to_dummies(cs.by_dtype([pl.Boolean, pl.Categorical, pl.Enum]))
    .with_columns(pl.all().cast(pl.Float32))
    .to_jax()
)

# Extract outcome
outcome = data["outcome"].to_jax()
censored = data["censored"].to_jax()
censoring_time = data["censoring_time"].to_jax()

# Write to disk
jnp.savez("data/array_data.npz", predictors, outcome, censored, censoring_time)
