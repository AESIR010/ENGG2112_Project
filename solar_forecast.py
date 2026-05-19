from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 6 # dataset in 10 min intervals - 1hr ahead forecast
TARGET = "TARGET_FUTURE"

DATASET_FILE = "humeridge_2024_10min_dataset_WOKRING.csv"

LEAKAGE_COLUMNS = [
    "target_generation_wh",
    "target_generation_kwh",
    "target_average_power_w",
    "capacity_factor",

    "pv_interval_average_power_w",
    "pv_instantaneous_power_w",
    "pv_average_power_w",
    "pv_normalised_output_kw_per_kw",
]

def load_dataset():
    path = DATA_DIR / DATASET_FILE

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def create_future_target(df):
    df = df.copy()

    df["TARGET_FUTURE"] = (
        df["target_generation_wh"]
        .shift(-FORECAST_HORIZON)
    )

    return df

def add_lag_features(df):
    df = df.copy()

    lag_columns = [
        "sat_global_tilted_irradiance",
        "sat_shortwave_radiation",
        "weather_cloud_cover",
        "weather_temperature_2m",
    ]

    for col in lag_columns:

        df[f"{col}_lag_10min"] = df[col].shift(1)

        df[f"{col}_lag_1h"] = df[col].shift(6)

        df[f"{col}_lag_24h"] = df[col].shift(144)

    # lag previous power output
    df["power_lag_10min"] = (
        df["target_generation_wh"].shift(1)
    )

    df["power_lag_1h"] = (
        df["target_generation_wh"].shift(6)
    )

    df["power_lag_24h"] = (
        df["target_generation_wh"].shift(144)
    )

    return df

def add_rolling_features(df):
    df = df.copy()

    rolling_columns = [
        "sat_global_tilted_irradiance",
        "weather_cloud_cover",
        "target_generation_wh",
    ]

    for col in rolling_columns:

        df[f"{col}_rolling_mean_1h"] = (
            df[col]
            .shift(1)
            .rolling(window=6)
            .mean()
        )

        df[f"{col}_rolling_std_1h"] = (
            df[col]
            .shift(1)
            .rolling(window=6)
            .std()
        )

    return df

def add_ramp_features(df):
    df = df.copy()

    ramp_columns = [
        "sat_global_tilted_irradiance",
        "weather_cloud_cover",
        "target_generation_wh",
    ]

    for col in ramp_columns:

        df[f"{col}_delta"] = df[col].diff()

    return df

def parse_datetime_column(series):
    """
    Handles both formats:
    - 15-05-2020 00:00
    - 2020-05-15 00:00:00
    """
    text = series.astype(str)
    is_iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    parsed.loc[is_iso] = pd.to_datetime(
        text.loc[is_iso],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    parsed.loc[~is_iso] = pd.to_datetime(
        text.loc[~is_iso],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(text, dayfirst=True, errors="coerce"))

    if parsed.isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    return parsed

def add_time_features(df):
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["is_daylight"] = (
        (df["sat_shortwave_radiation"] > 0) |
        (df["hour"].between(6, 18))
    ).astype(int)

    return df

def prepare_dataset():

    df = load_dataset()

    df = create_future_target(df)

    df = add_time_features(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    df = add_ramp_features(df)

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    return df


def chronological_split(df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """
    Time-aware split:
    first 60% timestamps = training
    next 20% timestamps = validation
    final 20% timestamps = testing
    """
    if round(train_ratio + val_ratio + test_ratio, 5) != 1:
        raise ValueError("Split ratios must add to 1.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_times = np.array(sorted(df["timestamp"].unique()))

    train_end = int(len(unique_times) * train_ratio)
    val_end = int(len(unique_times) * (train_ratio + val_ratio))

    train_times = set(unique_times[:train_end])
    val_times = set(unique_times[train_end:val_end])
    test_times = set(unique_times[val_end:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    val_df = df[df["timestamp"].isin(val_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    return train_df, val_df, test_df


EXCLUDED_COLUMNS = [
    "timestamp",
    TARGET,
]


def make_features(df):

    feature_columns = [
        c for c in df.columns
        if c not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()

    y = df[TARGET].copy()

    return X, y, feature_columns


def build_models():
    models = {
        "dummy_mean_baseline": DummyRegressor(strategy="mean"),

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "random_forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42
        )
    }

    return models


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def train_and_validate_models(X_train, y_train, X_val, y_val):
    models = build_models()

    trained_models = {}
    metric_rows = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_metrics = calculate_metrics(y_train, train_pred)
        val_metrics = calculate_metrics(y_val, val_pred)

        metric_rows.append({
            "model": model_name,
            "split": "train",
            **train_metrics
        })

        metric_rows.append({
            "model": model_name,
            "split": "validation",
            **val_metrics
        })

    metrics_df = pd.DataFrame(metric_rows)

    validation_results = metrics_df[metrics_df["split"] == "validation"]
    best_model_name = validation_results.sort_values("RMSE").iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, metrics_df


def plot_actual_vs_predicted(predictions_df):
    plt.figure(figsize=(12, 6))

    plot_df = predictions_df.sort_values("timestamp")

    plt.plot(
        plot_df["timestamp"],
        plot_df["actual"],
        label="Actual"
    )

    plt.plot(
        plot_df["timestamp"],
        plot_df["predicted"],
        label="Predicted"
    )

    plt.xlabel("Date/time")
    plt.ylabel("Plant AC power output")
    plt.title("Actual vs Predicted Solar Plant Output")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_residuals(predictions_df):
    residuals = predictions_df["actual_AC_POWER"] - predictions_df["predicted_AC_POWER"]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions_df["predicted_AC_POWER"],
        residuals,
        alpha=0.6
    )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted AC power output")
    plt.ylabel("Residual: actual - predicted")
    plt.title("Residual Plot")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
    plt.close()


def main():
    print("Preparing dataset...")
    df = prepare_dataset()

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    df.to_csv(OUTPUT_DIR / "processed_solar_dataset.csv", index=False)

    print(f"Total prepared rows: {len(df)}")

    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Testing rows: {len(test_df)}")

    X_train, y_train, feature_columns = make_features(train_df)

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    best_model_name, best_model, metrics_df = train_and_validate_models(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print(f"\nBest model selected by validation RMSE: {best_model_name}")

    test_predictions = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\nFinal test performance:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    test_row = {
        "model": best_model_name,
        "split": "test",
        **test_metrics
    }

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([test_row])],
        ignore_index=True
    )

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    predictions_df = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "actual": y_test,
        "predicted": test_predictions
    })

    predictions_df["error"] = (
        predictions_df["actual"] -
        predictions_df["predicted"]
    )

    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    plot_actual_vs_predicted(predictions_df)
    plot_residuals(predictions_df)

    model_bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "encoded_feature_columns": encoded_features,
        "target": TARGET,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "test_metrics": test_metrics
    }

    joblib.dump(model_bundle, MODEL_DIR / "best_solar_model.joblib")

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'processed_solar_dataset.csv'}")
    print(f"- {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'actual_vs_predicted.png'}")
    print(f"- {OUTPUT_DIR / 'residual_plot.png'}")
    print(f"- {MODEL_DIR / 'best_solar_model.joblib'}")


if __name__ == "__main__":
    main()