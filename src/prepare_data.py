from __future__ import annotations

import heapq
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import holidays as hol
import numpy as np
import pandas as pd
import pytz
import requests

# =========================================================
# CONFIG
# =========================================================

TZ_NAME = "Europe/Vienna"
TZ_LOCAL = pytz.timezone(TZ_NAME)
TZ_UTC = pytz.UTC
FREQ = "15min"

ENTSOE_BASE_URL = "https://transparency.entsoe.eu"
ENTSOE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

EEX_BASE_URL = "https://api.eex-group.com/pub/transparency/non-availability-events/Power"
EEX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.eex-transparency.com",
    "Referer": "https://www.eex-transparency.com/",
}

MAX_RETRIES = 5
CHUNK_DAYS = 6
AUSTRIA_AREA = "10YAT-APG------L"
ROR_LABEL = "Hydro Run-of-river and poundage"

FINAL_COLUMNS = [
    "datetime",
    "id1",
    "day_ahead_price",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "free_day",
    "load_actual_mw",
    "load_delta",
    "wind_delta",
    "solar_delta",
    "import_delta",
    "export_delta",
    "net_import_total",
    "fossil_gas_mw",
    "biomass_mw",
    "hydro_run_of_river_and_poundage_mw",
    "hydro_water_reservoir_mw",
    "hydro_pumped_storage_mw",
    "solar_mw",
    "wind_onshore_mw",
    "outage_total_true",
    "ramp_outage",
]

EXCHANGE_COUNTRIES = ["DE", "CH", "CZ", "IT", "HU", "SI"]

# =========================================================
# PATHS
# =========================================================

def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "src").exists() and ((path / "data_final").exists() or (path / "notebooks").exists()):
            return path
    return current


def default_data_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return root / "data_final" / "combined_data_qh.csv"

def default_price_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return root / "data_final" / "prices" / "id1_id3.xlsx"

# =========================================================
# TIME HELPERS
# =========================================================

def _localize_start(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str).tz_localize(TZ_NAME)


def _localize_end_exclusive(date_str: str) -> pd.Timestamp:
    # User-facing end date is inclusive, so 2026-01-31 means until 2026-02-01 00:00 exclusive.
    return pd.Timestamp(date_str).tz_localize(TZ_NAME) + pd.Timedelta(days=1)


def _required_index(date_from: str, date_to: str) -> pd.DatetimeIndex:
    start = _localize_start(date_from)
    end_excl = _localize_end_exclusive(date_to)
    return pd.date_range(start, end_excl - pd.Timedelta(minutes=15), freq=FREQ)


def _date_str(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ_NAME).strftime("%Y-%m-%d")


def _missing_spans(existing: pd.DataFrame, date_from: str, date_to: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    required = _required_index(date_from, date_to)
    if existing.empty or "datetime" not in existing.columns:
        return [(required[0], required[-1])]

    have = pd.DatetimeIndex(pd.to_datetime(existing["datetime"], utc=True).dt.tz_convert(TZ_NAME))
    missing = required.difference(have)
    if missing.empty:
        return []

    spans = []
    start = prev = missing[0]
    for ts in missing[1:]:
        if ts - prev == pd.Timedelta(minutes=15):
            prev = ts
        else:
            spans.append((start, prev))
            start = prev = ts
    spans.append((start, prev))
    return spans

# =========================================================
# ENTSO-E PARSERS
# =========================================================

def _parse_iso_duration(res: str) -> pd.Timedelta:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", res)
    if not m:
        raise ValueError(f"Unsupported duration: {res}")
    return pd.Timedelta(hours=int(m.group(1) or 0), minutes=int(m.group(2) or 0))


def _iterate_periods(data: dict):
    for inst in data.get("instanceList", []):
        for period in inst.get("curveData", {}).get("periodList", []):
            start_utc = pd.to_datetime(period["timeInterval"]["from"], utc=True)
            res = _parse_iso_duration(period.get("resolution", "PT15M"))
            for idx_str, vals in period.get("pointMap", {}).items():
                try:
                    idx = int(idx_str)
                except Exception:
                    continue
                yield inst, start_utc + idx * res, vals


def parse_meta_columns(data: dict) -> pd.DataFrame:
    meta = data.get("metaData", [])
    col_names = [m.get("code", f"col{i}") for i, m in enumerate(meta)]
    rows = []
    for _, ts, vals in _iterate_periods(data):
        if not isinstance(vals, list):
            vals = [vals]
        vals = vals + [None] * (len(col_names) - len(vals))
        row = {"timestamp_utc": ts}
        row.update({c: v for c, v in zip(col_names, vals)})
        rows.append(row)
    return pd.DataFrame(rows)


def parse_load(data: dict) -> pd.DataFrame:
    rows = []
    for _, ts, vals in _iterate_periods(data):
        vals = vals if isinstance(vals, list) else [vals]
        rows.append({
            "timestamp_utc": ts,
            "load_forecast_mw": vals[0] if len(vals) > 0 else None,
            "load_actual_mw": vals[1] if len(vals) > 1 else None,
        })
    return pd.DataFrame(rows)


def parse_generation_by_type(data: dict) -> pd.DataFrame:
    mapping = {
        "B01": "biomass_mw",
        "B04": "fossil_gas_mw",
        "B10": "hydro_pumped_storage_mw",
        "B11": "hydro_run_of_river_and_poundage_mw",
        "B12": "hydro_water_reservoir_mw",
        "B16": "solar_mw",
        "B19": "wind_onshore_mw",
    }
    rows = []
    for inst, ts, vals in _iterate_periods(data):
        prod_type = inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE", "UNKNOWN")
        val = vals[0] if isinstance(vals, list) and vals else vals
        if isinstance(val, dict):
            val = val.get("quantity")
        rows.append({
            "timestamp_utc": ts,
            "PRODUCTION_TYPE": mapping.get(prod_type, prod_type),
            "value": val,
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc"])
    return pd.DataFrame(rows).pivot(index="timestamp_utc", columns="PRODUCTION_TYPE", values="value").reset_index()


def parse_scheduled_exchange(data: dict) -> pd.DataFrame:
    contract_map = {"A01": "day_ahead", "A05": "total"}
    area_map = {
        "BZN|10YAT-APG------L": "AT",
        "BZN|10YCH-SWISSGRIDZ": "CH",
        "BZN|10YCZ-CEPS-----N": "CZ",
        "BZN|10Y1001A1001A82H": "DE",
        "BZN|10YHU-MAVIR----U": "HU",
        "BZN|10Y1001A1001A73I": "IT",
        "BZN|10YSI-ELES-----O": "SI",
    }
    rows = []
    for inst, ts, vals in _iterate_periods(data):
        dims = inst.get("businessDimensionMap", {})
        out_area = area_map.get(dims.get("OUT_AREA"), dims.get("OUT_AREA"))
        in_area = area_map.get(dims.get("IN_AREA"), dims.get("IN_AREA"))
        contract = contract_map.get(dims.get("CONTRACT_TYPE"), dims.get("CONTRACT_TYPE"))
        val = vals[0] if isinstance(vals, list) and vals else None
        if in_area and out_area and contract:
            rows.append({"timestamp_utc": ts, f"{in_area}_to_{out_area}_{contract}": val})
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc"])
    return pd.DataFrame(rows).groupby("timestamp_utc", as_index=False).agg("first")


def parse_day_ahead_prices(data: dict) -> pd.DataFrame:
    rows = []
    for _, ts, vals in _iterate_periods(data):
        rows.append({"timestamp_utc": ts, "day_ahead_price": vals[0] if isinstance(vals, list) and vals else None})
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "day_ahead_price"])
    return pd.DataFrame(rows).groupby("timestamp_utc", as_index=False).agg("first")


ENTSOE_ENDPOINTS = {
    "load": {"endpoint": "/load/total/dayAhead/load", "parser": parse_load},
    "solar_forecast": {"endpoint": "/generation/forecast/windAndSolar/solar/load", "parser": parse_meta_columns},
    "wind_onshore_forecast": {"endpoint": "/generation/forecast/windAndSolar/onshore/load", "parser": parse_meta_columns},
    "generation_by_type": {"endpoint": "/generation/actual/perType/generation/load", "parser": parse_generation_by_type},
    "scheduled_exchange": {"endpoint": "/transmission/scheduledExchange/load", "parser": parse_scheduled_exchange},
    "day_ahead_prices": {"endpoint": "/market/energyPrices/load", "parser": parse_day_ahead_prices},
}


def _post_with_retry(endpoint: str, payload: dict) -> dict:
    url = ENTSOE_BASE_URL + endpoint
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=ENTSOE_HEADERS, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"ENTSO-E attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}")
            time.sleep(5)
    raise RuntimeError("unreachable")


def _generate_query_window(date_from: str, date_to: str):
    local_start = TZ_LOCAL.localize(datetime.strptime(date_from, "%Y-%m-%d"))
    local_end = TZ_LOCAL.localize(datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1))
    utc_start = (local_start - timedelta(days=1)).astimezone(TZ_UTC)
    utc_end = (local_end + timedelta(days=1)).astimezone(TZ_UTC)
    return local_start, local_end, utc_start, utc_end


def _chunk_range(start: datetime, end: datetime, chunk_days: int):
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def query_entsoe(dataset: str, date_from: str, date_to: str, area: str = AUSTRIA_AREA, chunk_days: int = CHUNK_DAYS) -> pd.DataFrame:
    cfg = ENTSOE_ENDPOINTS[dataset]
    local_start, local_end, utc_start, utc_end = _generate_query_window(date_from, date_to)
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    dfs = []
    for chunk_start, chunk_end in _chunk_range(utc_start, utc_end, chunk_days):
        print(f"Querying {dataset}: {chunk_start} -> {chunk_end}")
        payload = {
            "dateTimeRange": {"from": chunk_start.strftime(fmt), "to": chunk_end.strftime(fmt)},
            "areaList": [f"BZN|{area}"],
            "timeZone": "UTC",
            "sorterList": [],
            "filterMap": {},
        }
        dfs.append(cfg["parser"](_post_with_retry(cfg["endpoint"], payload)))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["datetime"] = df["timestamp_utc"].dt.tz_convert(TZ_NAME)
    df = df.drop(columns=["timestamp_utc"], errors="ignore").set_index("datetime").sort_index()
    df = df[(df.index >= local_start) & (df.index < local_end)]
    return df.apply(pd.to_numeric, errors="coerce")


def query_entsoe_qh(date_from: str, date_to: str) -> pd.DataFrame:
    df_load = query_entsoe("load", date_from, date_to)
    df_solar = query_entsoe("solar_forecast", date_from, date_to)
    df_wind = query_entsoe("wind_onshore_forecast", date_from, date_to)
    df_generation = query_entsoe("generation_by_type", date_from, date_to)
    df_exchange = query_entsoe("scheduled_exchange", date_from, date_to)
    df_prices = query_entsoe("day_ahead_prices", date_from, date_to, chunk_days=1)

    df_solar = df_solar[[c for c in ["DAY_AHEAD", "CURRENT"] if c in df_solar.columns]].rename(
        columns={"DAY_AHEAD": "solar_day_ahead", "CURRENT": "solar_current"}
    )
    df_wind = df_wind[[c for c in ["DAY_AHEAD", "CURRENT"] if c in df_wind.columns]].rename(
        columns={"DAY_AHEAD": "wind_day_ahead", "CURRENT": "wind_current"}
    )
    df_exchange_qh = df_exchange.resample(FREQ).ffill()

    return (
        df_load.join(df_solar, how="outer")
        .join(df_wind, how="outer")
        .join(df_generation, how="outer")
        .join(df_prices, how="outer")
        .join(df_exchange_qh, how="outer")
        .sort_index()
    )

# =========================================================
# EEX OUTAGE DATA
# =========================================================

def fetch_outage_events(country: str, date_from: str, date_to: str) -> pd.DataFrame:
    params = {"country": country, "concernedDateFrom": date_from, "concernedDateTo": date_to}
    session = requests.Session()
    session.headers.update(EEX_HEADERS)
    r = session.get(EEX_BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    return pd.DataFrame([dict(zip(js["header"], row)) for row in js["data"]])


def split_message_id(message_id):
    try:
        base, version = str(message_id).rsplit("_", 1)
        return base, int(version)
    except Exception:
        return str(message_id), 0


def clean_outage_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    df = df_raw.copy()
    df = df[(df["unavailabilityType"] == "Unplanned") & (df["eventType"] == "Production unavailability")].copy()
    df["base_id"] = df["messageID"].map(lambda x: split_message_id(x)[0])
    df["version"] = df["messageID"].map(lambda x: split_message_id(x)[1])
    for c in ["eventStart", "eventStop", "modified"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert(TZ_NAME)
    for c in ["installedCapacity", "availableCapacity", "nonAvailableCapacity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["unit_id"] = df["facilityName"].fillna("").astype(str) + "|" + df["unitName"].fillna("").astype(str) + "|" + df["eIC"].fillna("").astype(str)
    df["fuel_group"] = np.where(df["fuelType"].eq(ROR_LABEL), "ror", "other")
    df["effective_outage"] = np.where(
        df["installedCapacity"].notna() & df["availableCapacity"].notna(),
        df["installedCapacity"] - df["availableCapacity"],
        df["nonAvailableCapacity"],
    )
    df["effective_outage"] = pd.to_numeric(df["effective_outage"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["eventStart", "eventStop", "modified"])
    return df[df["eventStop"] > df["eventStart"]].copy()


def latest_version_per_base(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
        return x.copy()
    idx = x.groupby("base_id")["version"].idxmax()
    return x.loc[idx].copy()


def build_unit_step_segments(events: pd.DataFrame) -> pd.DataFrame:
    seg_rows = []
    if events.empty:
        return pd.DataFrame(columns=["unit_id", "fuel_group", "start", "stop", "outage"])
    for (unit_id, fuel_group), g in events.groupby(["unit_id", "fuel_group"], sort=False):
        starts = g[["eventStart", "effective_outage"]].copy()
        starts["delta"] = 1
        stops = g[["eventStop", "effective_outage"]].copy()
        stops.columns = ["eventStart", "effective_outage"]
        stops["delta"] = -1
        changes = pd.concat([starts, stops], ignore_index=True).rename(columns={"eventStart": "time", "effective_outage": "cap"})
        changes = changes.sort_values(["time", "delta"])
        active_counts = defaultdict(int)
        max_heap = []
        grouped = {t: grp[["cap", "delta"]].to_records(index=False) for t, grp in changes.groupby("time", sort=True)}
        change_times = changes["time"].drop_duplicates().sort_values().to_list()
        for i, t in enumerate(change_times[:-1]):
            for rec in grouped[t]:
                cap = float(rec.cap)
                if int(rec.delta) == 1:
                    active_counts[cap] += 1
                    heapq.heappush(max_heap, -cap)
                else:
                    active_counts[cap] -= 1
            while max_heap and active_counts[-max_heap[0]] <= 0:
                heapq.heappop(max_heap)
            next_t = change_times[i + 1]
            current_outage = -max_heap[0] if max_heap else 0.0
            if next_t > t and current_outage > 0:
                seg_rows.append((unit_id, fuel_group, t, next_t, current_outage))
    return pd.DataFrame(seg_rows, columns=["unit_id", "fuel_group", "start", "stop", "outage"])


def aggregate_unit_segments_to_global(unit_segments: pd.DataFrame) -> pd.DataFrame:
    if unit_segments.empty:
        return pd.DataFrame(columns=["start", "stop", "outage_ror", "outage_other", "outage_total"])
    deltas = defaultdict(lambda: {"ror": 0.0, "other": 0.0})
    for row in unit_segments.itertuples():
        deltas[row.start][row.fuel_group] += row.outage
        deltas[row.stop][row.fuel_group] -= row.outage
    cur_ror = cur_other = 0.0
    rows = []
    times = sorted(deltas.keys())
    for i, t in enumerate(times[:-1]):
        cur_ror += deltas[t]["ror"]
        cur_other += deltas[t]["other"]
        next_t = times[i + 1]
        if next_t > t:
            rows.append((t, next_t, cur_ror, cur_other, cur_ror + cur_other))
    return pd.DataFrame(rows, columns=["start", "stop", "outage_ror", "outage_other", "outage_total"])


def sample_global_segments_to_grid(global_segments: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=time_index, columns=["outage_ror", "outage_other", "outage_total"], dtype=float)
    out[:] = 0.0
    if global_segments.empty:
        return out
    starts = global_segments["start"].to_numpy()
    stops = global_segments["stop"].to_numpy()
    grid = time_index.to_numpy()
    idx = np.searchsorted(starts, grid, side="right") - 1
    valid = (idx >= 0) & (grid < stops[np.clip(idx, 0, len(stops) - 1)])
    for col in ["outage_ror", "outage_other", "outage_total"]:
        vals = global_segments[col].to_numpy()
        arr = np.zeros(len(grid), dtype=float)
        arr[valid] = vals[idx[valid]]
        out[col] = arr
    return out


def query_outage_qh(date_from: str, date_to: str) -> pd.DataFrame:
    raw = fetch_outage_events("AT", date_from, date_to)
    df = clean_outage_events(raw)
    index = _required_index(date_from, date_to)
    latest = latest_version_per_base(df)
    truth_events = latest[(latest["eventStatus"] == "Active") & (latest["effective_outage"] > 0)].copy() if not latest.empty else latest
    unit_segments = build_unit_step_segments(truth_events)
    global_segments = aggregate_unit_segments_to_global(unit_segments)
    ts = sample_global_segments_to_grid(global_segments, index)
    ts = ts.rename(columns={
        "outage_ror": "outage_ror_true",
        "outage_other": "outage_other_true",
        "outage_total": "outage_total_true",
    })
    return ts

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def _safe_sum(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index)
    return df[present].sum(axis=1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)

    years = range(df.index.min().year, df.index.max().year + 1)
    holidays = set(hol.Austria(years=years).keys())
    df["free_day"] = ((df.index.weekday >= 5) | pd.Series(df.index.date, index=df.index).isin(holidays)).astype(int)

    import_day_ahead = _safe_sum(df, [f"{c}_to_AT_day_ahead" for c in EXCHANGE_COUNTRIES])
    export_day_ahead = _safe_sum(df, [f"AT_to_{c}_day_ahead" for c in EXCHANGE_COUNTRIES])
    import_total = _safe_sum(df, [f"{c}_to_AT_total" for c in EXCHANGE_COUNTRIES])
    export_total = _safe_sum(df, [f"AT_to_{c}_total" for c in EXCHANGE_COUNTRIES])

    df["net_import_total"] = import_total - export_total
    df["import_delta"] = import_total - import_day_ahead
    df["export_delta"] = export_total - export_day_ahead

    df["load_delta"] = df.get("load_actual_mw") - df.get("load_forecast_mw")
    df["wind_delta"] = df.get("wind_onshore_mw") - df.get("wind_day_ahead")
    df["solar_delta"] = df.get("solar_mw") - df.get("solar_day_ahead")

    if "outage_total_true" not in df.columns:
        df["outage_total_true"] = 0.0
    df["ramp_outage"] = df["outage_total_true"] - df["outage_total_true"].shift(1)
    df["ramp_outage"] = df["ramp_outage"].fillna(0.0)

    for col in FINAL_COLUMNS:
        if col != "datetime" and col not in df.columns:
            df[col] = np.nan

    return df.reset_index(names="datetime")[FINAL_COLUMNS]


def load_id1_prices_qh(
    date_from: str,
    date_to: str,
    price_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(price_path) if price_path is not None else default_price_path()

    if not path.exists():
        raise FileNotFoundError(
            f"ID1 price file not found: {path}\n"
            "Please place id1_id3.xlsx in data_final/prices/."
        )

    df_prices = pd.read_excel(
        path,
        sheet_name="qh",
        usecols="A:C",
        skiprows=5,
        header=None,
    )

    df_prices.columns = ["datetime", "id1", "id3"]

    df_prices["datetime"] = pd.to_datetime(df_prices["datetime"])

    df_prices["datetime"] = (
        df_prices["datetime"]
        .dt.tz_localize(
            TZ_NAME,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    )

    df_prices["datetime"] = pd.to_datetime(
        df_prices["datetime"],
        utc=True,
    ).dt.tz_convert(TZ_NAME)

    df_prices["id1"] = pd.to_numeric(df_prices["id1"], errors="coerce")

    required = _required_index(date_from, date_to)

    df_prices = (
        df_prices[["datetime", "id1"]]
        .dropna(subset=["id1"])
        .sort_values("datetime")
        .drop_duplicates(subset="datetime", keep="last")
    )

    missing = required.difference(pd.DatetimeIndex(df_prices["datetime"]))

    if not missing.empty:
        raise ValueError(
            "ID1 price file does not fully cover the requested period.\n"
            f"File: {path}\n"
            f"Requested: {date_from} to {date_to}\n"
            f"Missing timestamps: {len(missing):,}\n"
            f"First missing: {missing[0]}\n"
            f"Last missing: {missing[-1]}"
        )

    df_prices = df_prices[df_prices["datetime"].isin(required)]

    return df_prices.set_index("datetime")[["id1"]]


def build_increment(
    date_from: str,
    date_to: str,
    price_path: str | Path | None = None,
) -> pd.DataFrame:
    entsoe = query_entsoe_qh(date_from, date_to)
    outage = query_outage_qh(date_from, date_to)
    prices = load_id1_prices_qh(date_from, date_to, price_path=price_path)

    merged = (
        entsoe
        .join(outage, how="left")
        .join(prices, how="left")
    )

    return add_features(merged)

# =========================================================
# MAIN PUBLIC FUNCTION
# =========================================================

def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)
    return df


def save_combined(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ_NAME)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")

    # recompute ramp after combining, so first row of every increment is correct.
    if "outage_total_true" in df.columns:
        df["ramp_outage"] = df["outage_total_true"] - df["outage_total_true"].shift(1)
        df["ramp_outage"] = df["ramp_outage"].fillna(0.0)

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FINAL_COLUMNS]
    df.to_csv(path, index=False)


def prepare_all_data(
    date_from: str,
    date_to: str,
    data_path: str | Path | None = None,
    price_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Ensure that data_final/combined_data_qh.csv covers the requested inclusive date range.

    If timestamps are missing, only missing contiguous QH spans are queried,
    transformed into final features, appended, deduplicated, and written back.
    """
    path = Path(data_path) if data_path is not None else default_data_path()
    existing = load_existing(path)
    spans = _missing_spans(existing, date_from, date_to)

    if not spans:
        print(f"Data already covers {date_from} to {date_to}.")
        required = _required_index(date_from, date_to)
        mask = existing["datetime"].isin(required)
        return existing.loc[mask, FINAL_COLUMNS].sort_values("datetime").reset_index(drop=True)

    increments = []
    for start_ts, end_ts in spans:
        q_from = _date_str(start_ts)
        q_to = _date_str(end_ts)
        print(f"Missing span: {start_ts} -> {end_ts}. Querying {q_from} to {q_to}.")
        inc = build_increment(q_from, q_to, price_path=price_path)
        inc["datetime"] = pd.to_datetime(inc["datetime"], utc=True).dt.tz_convert(TZ_NAME)
        inc = inc[(inc["datetime"] >= start_ts) & (inc["datetime"] <= end_ts)]
        increments.append(inc)

    combined = pd.concat([existing, *increments], ignore_index=True)
    save_combined(combined, path)

    result = load_existing(path)
    required = _required_index(date_from, date_to)
    result = result[result["datetime"].isin(required)].sort_values("datetime").reset_index(drop=True)
    print(f"Saved combined data to: {path}")
    print(f"Returned rows: {len(result):,}")
    return result[FINAL_COLUMNS]
