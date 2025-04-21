import polars as pl

# Read in data
data = pl.read_excel("data/raw-data/synthetic_data_stats_competition_2025_final.xlsx")
data_dict = pl.read_excel(
    "data/raw-data/Data dictionary_stats competition 2025_final.xlsx", columns=[0, 1, 3]
)

# Convert datatype mapping to recognized Polars dtypes
dtype_mapping = {}
for name, dtype in data_dict.select("Variable name", "Variable type").iter_rows():
    match dtype:
        case "alpha_num":
            dtype_mapping[name] = pl.String
        case "boolean":
            dtype_mapping[name] = pl.Boolean
        case "numeric":
            dtype_mapping[name] = pl.Float64
        case _:
            pass

# Cast to proper datatypes
data = data.cast(dtype_mapping)

# Format sex column
data = data.with_columns(
    pl.col("demographics_birth_sex")
    .cast(pl.String)
    .replace({1: "male", 2: "female"})
    .cast(pl.Enum(["female", "male"]))
)

# Remove individuals who were not really followed up
data = data.filter(pl.col("follow_up_duration").ne(0.0))

# Remove predictors with no information
vars = [col for col in data.columns[0:-5] if data[col].n_unique() == 1]
data = data.drop(vars)
data_dict = data_dict.filter(~pl.col("Variable name").is_in(vars))

# Write to parquet file
data.write_parquet("data/ssc_competition_data.parquet")
data_dict.write_parquet("data/ssc_competition_data_dict.parquet")
