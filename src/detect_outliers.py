from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TZ_NAME = "Europe/Vienna"


@dataclass
class OutlierConfig:
    mad_threshold: float = 3.5
    iqr_multiplier: float = 1.5
    residual_col: str = "residual"


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    month: '2026-02' or '2026-02-01'
    returns selected month start and month end inclusive.
    """
    month_start = pd.Timestamp(month).tz_localize(TZ_NAME).replace(day=1)
    next_month = month_start + pd.DateOffset(months=1)
    month_end = next_month - pd.Timedelta(days=1)
    return month_start, month_end


def reference_bounds_for_month(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Previous 12 full months before selected month.
    Example:
    selected month: 2026-02
    reference: 2025-02-01 to 2026-01-31
    """
    month_start, _ = month_bounds(month)
    ref_start = month_start - pd.DateOffset(months=12)
    ref_end = month_start - pd.Timedelta(minutes=15)
    return ref_start, ref_end


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "src").exists() and (path / "data_final").exists():
            return path
    return current


def default_price_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return root / "data_final" / "prices" / "id1_id3.xlsx"


def load_id1_price_dates(
    price_path: str | Path | None = None,
    sheet_name: str = "qh",
) -> pd.DataFrame:
    path = Path(price_path) if price_path is not None else default_price_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}. "
            "Please place id1_id3.xlsx in data_final/prices/."
        )

    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols="A:C",
        skiprows=5,
        header=None,
    )

    df.columns = ["datetime", "id1", "id3"]

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize(
            TZ_NAME,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    )

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)
    df["id1"] = pd.to_numeric(df["id1"], errors="coerce")

    return df[["datetime", "id1"]].dropna(subset=["id1"])


def get_available_selected_month_end(
    month: str,
    price_path: str | Path | None = None,
) -> pd.Timestamp:
    """
    Checks how far ID1 prices are available within the selected month.
    """
    month_start, month_end = month_bounds(month)
    month_end_excl = month_end + pd.Timedelta(days=1)

    prices = load_id1_price_dates(price_path=price_path)

    prices_month = prices[
        (prices["datetime"] >= month_start)
        & (prices["datetime"] < month_end_excl)
    ].copy()

    if prices_month.empty:
        raise ValueError(
            f"No ID1 QH prices found for selected month {month_start:%Y-%m}."
        )

    return prices_month["datetime"].max()


def validate_selected_month(
    month: str,
    price_path: str | Path | None = None,
) -> dict:
    month_start, month_end = month_bounds(month)
    available_until = get_available_selected_month_end(
        month=month,
        price_path=price_path,
    )

    return {
        "month_start": month_start,
        "month_end": month_end,
        "available_until": available_until,
        "date_from_for_prepare": (month_start - pd.DateOffset(months=12)).strftime("%Y-%m-%d"),
        "date_to_for_prepare": available_until.strftime("%Y-%m-%d"),
        "selected_month": month_start.strftime("%Y-%m"),
    }


def fit_mad_thresholds(
    reference_residuals: pd.Series,
    threshold: float = 3.5,
) -> dict:
    x = pd.to_numeric(reference_residuals, errors="coerce").dropna().to_numpy()

    if len(x) == 0:
        raise ValueError("No valid reference residuals available for MAD.")

    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - median)))

    if mad == 0 or np.isnan(mad):
        raise ValueError("MAD is zero or NaN. Cannot calculate MAD flags.")

    lower = median - (threshold * mad / 0.6745)
    upper = median + (threshold * mad / 0.6745)

    return {
        "method": "MAD",
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "lower": lower,
        "upper": upper,
    }


def fit_iqr_thresholds(
    reference_residuals: pd.Series,
    multiplier: float = 1.5,
) -> dict:
    x = pd.to_numeric(reference_residuals, errors="coerce").dropna().to_numpy()

    if len(x) == 0:
        raise ValueError("No valid reference residuals available for IQR.")

    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return {
        "method": "IQR",
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "multiplier": multiplier,
        "lower": lower,
        "upper": upper,
    }


def apply_outlier_flags(
    residuals_df: pd.DataFrame,
    month: str,
    config: OutlierConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Uses previous 12 full months as reference period.
    Flags only the selected month.
    """
    config = config or OutlierConfig()

    df = residuals_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)

    month_start, month_end = month_bounds(month)
    selected_end_excl = month_end + pd.Timedelta(days=1)

    ref_start, ref_end = reference_bounds_for_month(month)
    ref_end_excl = ref_end + pd.Timedelta(minutes=15)

    ref_mask = (
        (df["datetime"] >= ref_start)
        & (df["datetime"] < ref_end_excl)
    )

    selected_mask = (
        (df["datetime"] >= month_start)
        & (df["datetime"] < selected_end_excl)
    )

    ref = df.loc[ref_mask, config.residual_col]

    mad_info = fit_mad_thresholds(
        reference_residuals=ref,
        threshold=config.mad_threshold,
    )

    iqr_info = fit_iqr_thresholds(
        reference_residuals=ref,
        multiplier=config.iqr_multiplier,
    )

    df["mad_flag"] = False
    df["iqr_flag"] = False
    df["any_outlier_flag"] = False

    selected_resid = df.loc[selected_mask, config.residual_col]

    df.loc[selected_mask, "mad_flag"] = (
        (selected_resid < mad_info["lower"])
        | (selected_resid > mad_info["upper"])
    )

    df.loc[selected_mask, "iqr_flag"] = (
        (selected_resid < iqr_info["lower"])
        | (selected_resid > iqr_info["upper"])
    )

    df["any_outlier_flag"] = df["mad_flag"] | df["iqr_flag"]

    info = {
        "selected_month": month_start.strftime("%Y-%m"),
        "selected_start": month_start,
        "selected_end": month_end,
        "reference_start": ref_start,
        "reference_end": ref_end,
        "n_reference": int(ref_mask.sum()),
        "n_selected": int(selected_mask.sum()),
        "n_mad_flags": int(df.loc[selected_mask, "mad_flag"].sum()),
        "n_iqr_flags": int(df.loc[selected_mask, "iqr_flag"].sum()),
        "mad": mad_info,
        "iqr": iqr_info,
    }

    return df, info


def get_days_for_selected_month(flagged_df: pd.DataFrame, month: str) -> list[pd.Timestamp]:
    df = flagged_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)

    month_start, month_end = month_bounds(month)
    month_end_excl = month_end + pd.Timedelta(days=1)

    selected = df[
        (df["datetime"] >= month_start)
        & (df["datetime"] < month_end_excl)
    ]

    return sorted(pd.to_datetime(selected["datetime"].dt.date).unique())


def filter_day(flagged_df: pd.DataFrame, day) -> pd.DataFrame:
    df = flagged_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)

    day_start = pd.Timestamp(day).tz_localize(TZ_NAME)
    day_end = day_start + pd.Timedelta(days=1)

    return df[
        (df["datetime"] >= day_start)
        & (df["datetime"] < day_end)
    ].copy()