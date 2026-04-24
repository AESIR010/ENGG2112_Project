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

TARGET = "AC_POWER"


GENERATION_FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_2_Generation_Data.csv",
]

WEATHER_FILES = [
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]


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
        raise ValueError("Some DATE_TIME values could not be parsed.")

    return parsed


def load_generation_data():
    frames = []

    for file in GENERATION_FILES:
        path = DATA_DIR / file
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No generation CSV files found.")

    generation = pd.concat(frames, ignore_index=True)
    generation["DATE_TIME"] = parse_datetime_column(generation["DATE_TIME"])

    # Raw generation data is inverter-level.
    # Aggregate inverter output into total plant output per timestamp.
    plant_generation = (
        generation
        .groupby(["PLANT_ID", "DATE_TIME"], as_index=False)
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
            TOTAL_YIELD=("TOTAL_YIELD", "sum"),
            INVERTER_COUNT=("SOURCE_KEY", "nunique")
        )
        .sort_values(["DATE_TIME", "PLANT_ID"])
        .reset_index(drop=True)
    )

    return plant_generation


def load_weather_data():
    frames = []

    for file in WEATHER_FILES:
        path = DATA_DIR / file
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No weather CSV files found.")

    weather = pd.concat(frames, ignore_index=True)
    weather["DATE_TIME"] = parse_datetime_column(weather["DATE_TIME"])

    plant_weather = (
        weather
        .groupby(["PLANT_ID", "DATE_TIME"], as_index=False)
        .agg(
            AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
            MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
            IRRADIATION=("IRRADIATION", "mean")
        )
        .sort_values(["DATE_TIME", "PLANT_ID"])
        .reset_index(drop=True)
    )

    return plant_weather


def build_dataset():
    generation = load_generation_data()
    weather = load_weather_data()

    df = pd.merge(
        generation,
        weather,
        on=["PLANT_ID", "DATE_TIME"],
        how="inner"
    )

    df = df.sort_values(["DATE_TIME", "PLANT_ID"]).reset_index(drop=True)

    return df


def add_time_features(df):
    df = df.copy()

    df["hour"] = df["DATE_TIME"].dt.hour
    df["minute"] = df["DATE_TIME"].dt.minute
    df["day_of_week"] = df["DATE_TIME"].dt.dayofweek
    df["month"] = df["DATE_TIME"].dt.month
    df["day_of_year"] = df["DATE_TIME"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["is_daylight"] = (
        (df["IRRADIATION"] > 0) |
        (df["hour"].between(6, 18))
    ).astype(int)

    return df


def add_lag_features(df):
    """
    Dataset interval is 15 minutes.
    1 row = 15 minutes
    4 rows = 1 hour
    96 rows = 24 hours
    """
    df = df.copy()
    df = df.sort_values(["PLANT_ID", "DATE_TIME"]).reset_index(drop=True)

    grouped = df.groupby("PLANT_ID", group_keys=False)

    df["ac_lag_15min"] = grouped["AC_POWER"].shift(1)
    df["ac_lag_1h"] = grouped["AC_POWER"].shift(4)
    df["ac_lag_24h"] = grouped["AC_POWER"].shift(96)

    df["irr_lag_15min"] = grouped["IRRADIATION"].shift(1)
    df["irr_lag_1h"] = grouped["IRRADIATION"].shift(4)
    df["irr_lag_24h"] = grouped["IRRADIATION"].shift(96)

    df["rolling_ac_mean_1h"] = grouped["AC_POWER"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )

    df["rolling_ac_mean_24h"] = grouped["AC_POWER"].transform(
        lambda s: s.shift(1).rolling(window=96, min_periods=1).mean()
    )

    df["rolling_irr_mean_1h"] = grouped["IRRADIATION"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )

    lag_columns = [
        "ac_lag_15min",
        "ac_lag_1h",
        "ac_lag_24h",
        "irr_lag_15min",
        "irr_lag_1h",
        "irr_lag_24h",
        "rolling_ac_mean_1h",
        "rolling_ac_mean_24h",
        "rolling_irr_mean_1h",
    ]

    df[lag_columns] = df[lag_columns].fillna(0)

    df = df.sort_values(["DATE_TIME", "PLANT_ID"]).reset_index(drop=True)

    return df


def prepare_dataset():
    df = build_dataset()
    df = add_time_features(df)
    df = add_lag_features(df)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

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

    df = df.sort_values(["DATE_TIME", "PLANT_ID"]).reset_index(drop=True)

    unique_times = np.array(sorted(df["DATE_TIME"].unique()))

    train_end = int(len(unique_times) * train_ratio)
    val_end = int(len(unique_times) * (train_ratio + val_ratio))

    train_times = set(unique_times[:train_end])
    val_times = set(unique_times[train_end:val_end])
    test_times = set(unique_times[val_end:])

    train_df = df[df["DATE_TIME"].isin(train_times)].copy()
    val_df = df[df["DATE_TIME"].isin(val_times)].copy()
    test_df = df[df["DATE_TIME"].isin(test_times)].copy()

    return train_df, val_df, test_df


FEATURE_COLUMNS = [
    "PLANT_ID",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",

    "hour",
    "minute",
    "day_of_week",
    "month",
    "day_of_year",

    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_daylight",

    "ac_lag_15min",
    "ac_lag_1h",
    "ac_lag_24h",

    "irr_lag_15min",
    "irr_lag_1h",
    "irr_lag_24h",

    "rolling_ac_mean_1h",
    "rolling_ac_mean_24h",
    "rolling_irr_mean_1h",
]


def make_features(df, feature_columns_after_encoding=None):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()

    # Treat plant ID as categorical, not a normal numeric variable.
    X = pd.get_dummies(X, columns=["PLANT_ID"], prefix="plant", dtype=float)

    if feature_columns_after_encoding is None:
        feature_columns_after_encoding = list(X.columns)
    else:
        X = X.reindex(columns=feature_columns_after_encoding, fill_value=0)

    return X, y, feature_columns_after_encoding


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

    plot_df = predictions_df.sort_values("DATE_TIME")

    plt.plot(
        plot_df["DATE_TIME"],
        plot_df["actual_AC_POWER"],
        label="Actual"
    )

    plt.plot(
        plot_df["DATE_TIME"],
        plot_df["predicted_AC_POWER"],
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

    X_train, y_train, encoded_features = make_features(train_df)
    X_val, y_val, _ = make_features(val_df, encoded_features)
    X_test, y_test, _ = make_features(test_df, encoded_features)

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

    predictions_df = test_df[["DATE_TIME", "PLANT_ID", TARGET]].copy()
    predictions_df = predictions_df.rename(columns={TARGET: "actual_AC_POWER"})
    predictions_df["predicted_AC_POWER"] = test_predictions
    predictions_df["error"] = (
        predictions_df["actual_AC_POWER"] -
        predictions_df["predicted_AC_POWER"]
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