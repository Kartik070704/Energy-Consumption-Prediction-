"""Train and compare models for the WattWise energy prediction project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "energy_consumption_real_10000.xlsx"
ARTIFACTS = ROOT / "artifacts"
TARGET = "Energy_Consumption_kW"
NUMERIC_FEATURES = [
    "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1",
    "Sub_metering_2", "Sub_metering_3", "Hour", "Year", "Month", "DayOfYear",
    "IsWeekend", "HourSin", "HourCos", "DaySin", "DayCos",
]
CATEGORICAL_FEATURES = ["DayOfWeek"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create time features while keeping the original measured inputs."""
    data = frame.copy()
    dt = pd.to_datetime(data["DateTime"])
    hour = dt.dt.hour + dt.dt.minute / 60
    day = dt.dt.dayofweek
    data["Year"] = dt.dt.year
    data["Month"] = dt.dt.month
    data["DayOfYear"] = dt.dt.dayofyear
    data["Hour"] = hour
    data["IsWeekend"] = (day >= 5).astype(int)
    data["HourSin"] = np.sin(2 * np.pi * hour / 24)
    data["HourCos"] = np.cos(2 * np.pi * hour / 24)
    data["DaySin"] = np.sin(2 * np.pi * day / 7)
    data["DayCos"] = np.cos(2 * np.pi * day / 7)
    data["DayOfWeek"] = data["DayOfWeek"].astype(str)
    return data[FEATURES]


def build_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("weekday", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name="Energy_Data")
    df = df.dropna(subset=[TARGET, "DateTime"]).sort_values("DateTime").reset_index(drop=True)
    X = prepare_features(df)
    y = df[TARGET].astype(float)

    # Preserve time order: the newest 20% is held out as unseen future data.
    split_at = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    cv = TimeSeriesSplit(n_splits=4)
    candidates = {
        "Ridge Regression": (build_pipeline(Ridge(), scale_numeric=True), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}),
        "Lasso Regression": (build_pipeline(Lasso(max_iter=20000), scale_numeric=True), {"model__alpha": [0.0001, 0.001, 0.01, 0.1]}),
        "Decision Tree": (build_pipeline(DecisionTreeRegressor(random_state=42), scale_numeric=False), {
            "model__max_depth": [6, 10, 16, None], "model__min_samples_leaf": [2, 5, 10],
        }),
        "Random Forest": (build_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1), scale_numeric=False), {
            "model__n_estimators": [180, 260, 360], "model__max_depth": [12, 20, None],
            "model__min_samples_leaf": [1, 2, 4], "model__max_features": [0.7, 1.0],
        }),
    }

    results, fitted = [], {}
    for name, (pipeline, param_distributions) in candidates.items():
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_distributions,
            n_iter=12 if name == "Random Forest" else min(10, int(np.prod([len(v) for v in param_distributions.values()]))),
            scoring="neg_root_mean_squared_error", cv=cv, random_state=42,
            n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        predictions = search.best_estimator_.predict(X_test)
        results.append({
            "model": name,
            "cv_rmse": float(-search.best_score_),
            **metrics(y_test, predictions),
            "best_params": search.best_params_,
        })
        fitted[name] = search.best_estimator_
        print(f"{name}: CV RMSE={-search.best_score_:.4f} | test RMSE={results[-1]['rmse']:.4f} | test R²={results[-1]['r2']:.4f}")

    # Choose from training-only time-series CV, keeping the final test set untouched.
    winner = min(results, key=lambda item: item["cv_rmse"])
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(fitted[winner["model"]], ARTIFACTS / "energy_model.joblib")
    report = {
        "project_name": "WattWise — Household Energy Consumption Predictor",
        "target": TARGET,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_period_start": str(pd.to_datetime(df.iloc[split_at]["DateTime"])),
        "selected_model": winner["model"],
        "selected_best_params": winner["best_params"],
        "selected_metrics": {k: winner[k] for k in ("mae", "rmse", "r2", "cv_rmse")},
        "model_comparison": sorted(results, key=lambda item: item["cv_rmse"]),
        "features": FEATURES,
        "missing_values_dropped": int(pd.read_excel(DATA_PATH, sheet_name="Energy_Data").isna().any(axis=1).sum()),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSelected model: {winner['model']} (lowest training CV RMSE)")
    print(f"Artifacts saved in {ARTIFACTS}")


if __name__ == "__main__":
    main()
