import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.prepare_data import prepare_all_data
from src.predict_residuals import predict_residuals
from src.detect_outliers import (
    OutlierConfig,
    apply_outlier_flags,
    filter_day,
    get_days_for_selected_month,
    validate_selected_month,
)


st.set_page_config(
    page_title="Anomaly Explorer",
    layout="wide",
)

PRICE_PATH = Path("data_final/prices/id1_id3.xlsx")

st.title("Anomaly Explorer (ID1 price index)")


# =========================================================
# CACHE
# =========================================================

@st.cache_data(show_spinner=False)
def cached_predict_residuals(
    df: pd.DataFrame,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    return predict_residuals(
        df=df,
        date_from=date_from,
        date_to=date_to,
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Time period selection")

default_date = dt.date(dt.date.today().year, 1, 1)

selected_date = st.sidebar.date_input(
    "Select month",
    value=default_date,
)

selected_month = selected_date.strftime("%Y-%m")

st.sidebar.info(f"Selected month: {selected_month}")


# =========================================================
# LOAD DATA
# =========================================================

if st.sidebar.button("Load data", type="primary"):
    with st.spinner("Checking price file and loading source data..."):
        month_info = validate_selected_month(
            selected_month,
            price_path=PRICE_PATH,
        )

        df = prepare_all_data(
            date_from=month_info["date_from_for_prepare"],
            date_to=month_info["date_to_for_prepare"],
            price_path=PRICE_PATH,
        )

        st.session_state["selected_month"] = selected_month
        st.session_state["month_info"] = month_info
        st.session_state["df"] = df

        # Reset downstream state if another month is loaded
        st.session_state.pop("residuals_df", None)
        st.session_state.pop("flagged_df", None)
        st.session_state.pop("outlier_info", None)
        st.session_state.pop("selected_day", None)

    st.sidebar.success("Data has been loaded.")


if "month_info" in st.session_state:
    month_info = st.session_state["month_info"]

    st.sidebar.write(
        "Available ID1 until:",
        month_info["available_until"].strftime("%Y-%m-%d %H:%M"),
    )


# =========================================================
# OUTLIER SETTINGS
# =========================================================

st.sidebar.header("Outlier settings")

mad_threshold = st.sidebar.number_input(
    "MAD threshold",
    min_value=0.1,
    value=3.5,
    step=0.1,
)

iqr_multiplier = st.sidebar.number_input(
    "IQR multiplier",
    min_value=0.1,
    value=2.0,
    step=0.1,
)


# =========================================================
# FLAG OUTLIERS
# =========================================================

if st.sidebar.button("Flag outliers"):
    if "df" not in st.session_state:
        st.sidebar.warning("Please load data first.")
    else:
        with st.spinner("Predicting residuals and flagging outliers..."):
            df = st.session_state["df"]
            month_info = st.session_state["month_info"]
            selected_month_loaded = st.session_state["selected_month"]

            residuals_df = cached_predict_residuals(
                df=df,
                date_from=month_info["month_start"].strftime("%Y-%m-%d"),
                date_to=month_info["available_until"].strftime("%Y-%m-%d"),
            )

            flagged_df, outlier_info = apply_outlier_flags(
                residuals_df=residuals_df,
                month=selected_month_loaded,
                config=OutlierConfig(
                    mad_threshold=mad_threshold,
                    iqr_multiplier=iqr_multiplier,
                ),
            )

            st.session_state["residuals_df"] = residuals_df
            st.session_state["flagged_df"] = flagged_df
            st.session_state["outlier_info"] = outlier_info
            st.session_state.pop("selected_day", None)

        st.sidebar.success("Outliers have been flagged.")


# =========================================================
# MAIN PAGE
# =========================================================

if "flagged_df" not in st.session_state:
    st.info("Select a month in the sidebar, load the data, then flag outliers.")
    st.stop()


flagged_df = st.session_state["flagged_df"]
outlier_info = st.session_state["outlier_info"]
selected_month_loaded = st.session_state["selected_month"]

st.header("Results")


# =========================================================
# OUTLIER SUMMARY
# =========================================================

selected_mask = flagged_df["is_requested_period"]

mad_flags = int((flagged_df["mad_flag"] & selected_mask).sum())
iqr_flags = int((flagged_df["iqr_flag"] & selected_mask).sum())

both_flags = int(
    (
        flagged_df["mad_flag"]
        & flagged_df["iqr_flag"]
        & selected_mask
    ).sum()
)

any_flags = int(
    (
        (flagged_df["mad_flag"] | flagged_df["iqr_flag"])
        & selected_mask
    ).sum()
)

n_selected = int(selected_mask.sum())

mad_overlap_share = both_flags / mad_flags if mad_flags > 0 else 0.0
iqr_overlap_share = both_flags / iqr_flags if iqr_flags > 0 else 0.0
any_flag_share = any_flags / n_selected if n_selected > 0 else 0.0

summary_df = pd.DataFrame(
    [
        {
            "Method": "MAD",
            "Flags": mad_flags,
            "Share also flagged by other method": f"{mad_overlap_share:.1%}",
        },
        {
            "Method": "IQR",
            "Flags": iqr_flags,
            "Share also flagged by other method": f"{iqr_overlap_share:.1%}",
        },
    ]
)

st.subheader("Outlier summary")

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)

st.metric(
    "Share of selected month flagged by at least one method",
    f"{any_flag_share:.1%}",
)

st.write(
    f"Reference period: "
    f"{outlier_info['reference_start'].strftime('%Y-%m-%d')} "
    f"to {outlier_info['reference_end'].strftime('%Y-%m-%d')}"
)


# =========================================================
# DAY SELECTION
# =========================================================

st.subheader("Daily ID1 price plot")

days = get_days_for_selected_month(
    flagged_df,
    selected_month_loaded,
)

if not days:
    st.warning("No days available for the selected month.")
    st.stop()

flag_counts = []

for day in days:
    tmp = filter_day(flagged_df, day)
    flag_counts.append(
        {
            "day": day,
            "n_flags": int(tmp["any_outlier_flag"].sum()),
        }
    )

flag_counts_df = pd.DataFrame(flag_counts)

flag_counts_df = flag_counts_df.sort_values(
    ["n_flags", "day"],
    ascending=[False, True],
).reset_index(drop=True)

default_day = flag_counts_df.iloc[0]["day"]

if "selected_day" not in st.session_state:
    st.session_state["selected_day"] = default_day

if st.session_state["selected_day"] not in days:
    st.session_state["selected_day"] = default_day

current_idx = days.index(st.session_state["selected_day"])

col_prev, col_jump, col_next = st.columns([1, 4, 1])

with col_prev:
    if st.button("⬅ Previous day"):
        if current_idx > 0:
            st.session_state["selected_day"] = days[current_idx - 1]
            st.rerun()

with col_next:
    if st.button("Next day ➡"):
        if current_idx < len(days) - 1:
            st.session_state["selected_day"] = days[current_idx + 1]
            st.rerun()

with col_jump:
    selected_day = st.selectbox(
        "Jump to day",
        options=days,
        index=days.index(st.session_state["selected_day"]),
        format_func=lambda x: (
            f"{pd.Timestamp(x).strftime('%Y-%m-%d')} "
            f"(flags: "
            f"{int(flag_counts_df.loc[flag_counts_df['day'] == x, 'n_flags'].iloc[0])})"
        ),
    )

st.session_state["selected_day"] = selected_day

day_df = filter_day(flagged_df, selected_day)

mad_only = day_df[day_df["mad_flag"] & ~day_df["iqr_flag"]]
iqr_only = day_df[day_df["iqr_flag"] & ~day_df["mad_flag"]]
both = day_df[day_df["mad_flag"] & day_df["iqr_flag"]]

st.caption(
    f"Selected day flags: "
    f"{int(day_df['any_outlier_flag'].sum())} total "
    f"({int(day_df['mad_flag'].sum())} MAD, "
    f"{int(day_df['iqr_flag'].sum())} IQR)"
)


# =========================================================
# DAILY PLOT
# =========================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=day_df["datetime"],
        y=day_df["id1"],
        mode="lines",
        name="Actual ID1",
    )
)

fig.add_trace(
    go.Scatter(
        x=mad_only["datetime"],
        y=mad_only["id1"],
        mode="markers",
        name="MAD only",
        marker=dict(symbol="x", size=10),
    )
)

fig.add_trace(
    go.Scatter(
        x=iqr_only["datetime"],
        y=iqr_only["id1"],
        mode="markers",
        name="IQR only",
        marker=dict(symbol="circle-open", size=10),
    )
)

fig.add_trace(
    go.Scatter(
        x=both["datetime"],
        y=both["id1"],
        mode="markers",
        name="MAD + IQR",
        marker=dict(symbol="star", size=13),
    )
)

fig.update_layout(
    title=(
        f"ID1 price with outlier flags — "
        f"{pd.Timestamp(selected_day).strftime('%Y-%m-%d')}"
    ),
    xaxis_title="Time",
    yaxis_title="ID1 [EUR/MWh]",
    hovermode="x unified",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# SELECTED DAY TABLE
# =========================================================

st.subheader("Selected day data")

st.dataframe(
    day_df[
        [
            "datetime",
            "id1",
            "day_ahead_price",
            "id1_hat",
            "residual",
            "mad_flag",
            "iqr_flag",
            "any_outlier_flag",
        ]
    ],
    use_container_width=True,
)


# =========================================================
# MODEL PERFORMANCE STATISTICS
# =========================================================

st.subheader("Prediction performance over full residual window")

perf_df = flagged_df.dropna(
    subset=["id1", "id1_hat", "day_ahead_price"]
).copy()


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    err = y_true - y_pred
    abs_err = err.abs()

    mae = abs_err.mean()
    rmse = (err.pow(2).mean()) ** 0.5

    ss_res = err.pow(2).sum()
    ss_tot = (y_true - y_true.mean()).pow(2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    q95_abs_error = abs_err.quantile(0.95)
    mean_q95_abs_error = abs_err[abs_err >= q95_abs_error].mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Mean Q95 abs. error": mean_q95_abs_error,
    }


model_metrics = regression_metrics(
    y_true=perf_df["id1"],
    y_pred=perf_df["id1_hat"],
)

day_ahead_metrics = regression_metrics(
    y_true=perf_df["id1"],
    y_pred=perf_df["day_ahead_price"],
)

metrics_df = pd.DataFrame(
    [
        {
            "Prediction": "Model ID1_hat",
            **model_metrics,
        },
        {
            "Prediction": "Day-ahead price",
            **day_ahead_metrics,
        },
    ]
)

for col in ["MAE", "RMSE", "Mean Q95 abs. error"]:
    metrics_df[col] = metrics_df[col].round(2)

metrics_df["R²"] = metrics_df["R²"].round(3)

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# PERFORMANCE FOR INSPECTED MONTH ONLY
# =========================================================

st.subheader("Prediction performance for inspected month only")

month_perf_df = perf_df[
    perf_df["is_requested_period"]
].copy()

model_metrics_month = regression_metrics(
    y_true=month_perf_df["id1"],
    y_pred=month_perf_df["id1_hat"],
)

day_ahead_metrics_month = regression_metrics(
    y_true=month_perf_df["id1"],
    y_pred=month_perf_df["day_ahead_price"],
)

metrics_month_df = pd.DataFrame(
    [
        {
            "Prediction": "Model ID1_hat",
            **model_metrics_month,
        },
        {
            "Prediction": "Day-ahead price",
            **day_ahead_metrics_month,
        },
    ]
)

for col in ["MAE", "RMSE", "Mean Q95 abs. error"]:
    metrics_month_df[col] = metrics_month_df[col].round(2)

metrics_month_df["R²"] = metrics_month_df["R²"].round(3)

st.dataframe(
    metrics_month_df,
    use_container_width=True,
    hide_index=True,
)