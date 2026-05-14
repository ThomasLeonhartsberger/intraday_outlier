from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd


TZ_NAME = "Europe/Vienna"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    for path in [current, *current.parents]:
        if (path / "src").exists() and (path / "models").exists():
            return path

    return current


def _to_vienna_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ_NAME)


def get_residual_window(
    date_from: str,
    date_to: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    requested_start = pd.Timestamp(date_from).tz_localize(TZ_NAME)
    requested_end = pd.Timestamp(date_to).tz_localize(TZ_NAME) + pd.Timedelta(days=1)

    month_start = requested_start.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    residual_start = month_start - pd.DateOffset(months=12)

    return residual_start, requested_end


def get_model_year(date_from: str) -> int:
    return pd.Timestamp(date_from).year - 2


def get_default_model_path(date_from: str) -> Path:
    root = find_project_root()
    model_year = get_model_year(date_from)

    return root / "models" / f"nn_{model_year}_model.pkl"


def add_lag_features_from_model(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = _to_vienna_datetime(df["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)

    for col in feature_cols:
        match = re.match(r"(.+)_lag(\d+)$", col)

        if match is None:
            continue

        base_col = match.group(1)
        lag = int(match.group(2))

        if base_col not in df.columns:
            raise KeyError(
                f"Model needs lag feature '{col}', "
                f"but base column '{base_col}' is missing."
            )

        df[col] = df[base_col].shift(lag)

    return df


def _predict_model(model, X):
    try:
        return model.predict(X, verbose=0).ravel()
    except TypeError:
        return model.predict(X).ravel()


def predict_residuals(
    df: pd.DataFrame,
    date_from: str,
    date_to: str,
    model_path: str | Path | None = None,
    target_col: str = "id1",
) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = _to_vienna_datetime(df["datetime"])

    residual_start, residual_end_excl = get_residual_window(
        date_from=date_from,
        date_to=date_to,
    )

    requested_start = pd.Timestamp(date_from).tz_localize(TZ_NAME)
    requested_end_excl = pd.Timestamp(date_to).tz_localize(TZ_NAME) + pd.Timedelta(days=1)

    if model_path is None:
        model_path = get_default_model_path(date_from)
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)

    model = artifact["model"]
    scaler = artifact.get("scaler")
    bias_correction = artifact.get("bias_correction", 0.0)
    feature_cols = artifact["feature_cols"]

    # Recreate lag features required by the exported model.
    df = add_lag_features_from_model(
        df=df,
        feature_cols=feature_cols,
    )

    required_cols = [target_col, "day_ahead_price", *feature_cols]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for prediction: {missing_cols}")

    pred_df = df[
        (df["datetime"] >= residual_start)
        & (df["datetime"] < residual_end_excl)
    ].copy()

    pred_df = pred_df.dropna(subset=required_cols).copy()

    pred_df = pred_df.sort_values("datetime").reset_index(drop=True)

    X = pred_df[feature_cols]

    if scaler is not None:
        X_model = scaler.transform(X)
    else:
        X_model = X

    spread_hat_raw = _predict_model(model, X_model)
    spread_hat = spread_hat_raw + bias_correction

    actual_id1 = pred_df[target_col].to_numpy()
    day_ahead = pred_df["day_ahead_price"].to_numpy()

    spread_actual = actual_id1 - day_ahead
    id1_hat = day_ahead + spread_hat
    residual = actual_id1 - id1_hat

    out = pd.DataFrame({
        "datetime": pred_df["datetime"],
        "id1": actual_id1,
        "day_ahead_price": day_ahead,
        "spread_actual": spread_actual,
        "spread_hat_raw": spread_hat_raw,
        "spread_hat": spread_hat,
        "id1_hat": id1_hat,
        "residual": residual,
        "abs_residual": np.abs(residual),
    })

    out["model_year"] = get_model_year(date_from)
    out["model_path"] = str(model_path)

    out["is_requested_period"] = (
        (out["datetime"] >= requested_start)
        & (out["datetime"] < requested_end_excl)
    )

    return out