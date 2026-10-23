"""WattWise: an interactive energy analytics and prediction website."""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import DATA_PATH, TARGET, prepare_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "energy_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COLORS = {"teal": "#189b7c", "blue": "#587de0", "violet": "#8b65bc", "amber": "#d18a2c", "red": "#cc5965"}

st.set_page_config(
    page_title="WattWise | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#f4f8f7; --panel:#ffffff; --panel2:#f6faf9; --line:rgba(39,75,86,.13); --ink:#18323d; --muted:#71848d; --teal:#168e75; --blue:#587de0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 84% 0%,rgba(92,206,175,.12),transparent 27%),radial-gradient(ellipse at 4% 34%,rgba(131,164,235,.08),transparent 29%),linear-gradient(145deg,#f7faf9 0%,#f1f7f5 55%,#f8fafb 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#ffffff,#f4f8f7); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#24414b; }
.block-container { max-width:1510px; padding-top:1.25rem; padding-bottom:3rem; }
.brandbar { display:flex; align-items:center; justify-content:space-between; margin:0 0 18px; }
.brand { display:flex; align-items:center; gap:11px; font:800 20px Manrope,sans-serif; letter-spacing:-.04em; color:#17333e; }
.brand-icon { width:34px; height:34px; display:grid; place-items:center; color:#ffffff; font-size:19px; border-radius:11px; background:linear-gradient(135deg,#20ad8d,#63a9e9); box-shadow:0 8px 22px rgba(30,153,123,.18); }
.brand-note { color:#758993; font-size:10px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-top:3px; }
.live-badge { color:#16765f; font-size:11px; font-weight:700; padding:7px 11px; border:1px solid rgba(31,146,116,.2); border-radius:99px; background:rgba(49,178,143,.08); }
.hero { position:relative; overflow:hidden; border:1px solid rgba(81,139,155,.17); border-radius:22px; padding:27px 31px 26px; margin:0 0 21px; background:radial-gradient(circle at 86% 55%,rgba(80,194,156,.16),transparent 18%),linear-gradient(108deg,rgba(255,255,255,.98),rgba(240,249,246,.97) 62%,rgba(236,248,247,.96)); box-shadow:0 16px 40px rgba(37,73,76,.075); }
.hero:after { content:' '; position:absolute; width:210px;height:210px;right:-72px;top:-114px;border:1px solid rgba(60,162,141,.2);border-radius:50%;box-shadow:0 0 0 24px rgba(55,158,135,.025),0 0 0 50px rgba(55,158,135,.02); }
.eyebrow { color:var(--teal); font-size:10px; font-weight:800; letter-spacing:.17em; text-transform:uppercase; }
.hero h1 { font:800 clamp(27px,3vw,39px) Manrope,sans-serif; letter-spacing:-.045em; margin:8px 0 5px; color:#18323d; }
.hero p { color:#617782; max-width:780px; margin:0; font-size:14px; line-height:1.6; }
.page-number { position:absolute; right:28px;bottom:22px;color:rgba(56,103,111,.58);font:700 12px Manrope,sans-serif;letter-spacing:.12em; }
div[data-testid="stSegmentedControl"] { margin:0 0 20px; }
div[data-testid="stSegmentedControl"] button { border-radius:11px !important; font-weight:700 !important; }
div[data-testid="stSegmentedControl"] button[aria-pressed="true"] { box-shadow:0 4px 16px rgba(66,226,194,.12); }
.metric-card { height:118px; padding:16px 18px; border-radius:16px; border:1px solid rgba(54,91,99,.13); background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(248,251,250,.97)); box-shadow:0 9px 25px rgba(31,66,69,.055); }
.metric-top { display:flex; justify-content:space-between; align-items:center; color:#778b94; font-size:10px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.metric-icon { width:26px;height:26px;display:grid;place-items:center;border-radius:8px;color:#16866f;background:rgba(41,166,133,.1);font-size:13px; }
.metric-value { color:#193540; font:800 clamp(19px,2vw,25px) Manrope,sans-serif; margin-top:11px; letter-spacing:-.035em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.metric-note { color:#788c95; font-size:11px; margin-top:3px; }
.section-head { display:flex; align-items:end; justify-content:space-between; margin:23px 0 10px; }
.section-head h2 { color:#193540; font:750 19px Manrope,sans-serif; letter-spacing:-.025em; margin:0; }
.section-head p { color:#70838d; font-size:12px; margin:4px 0 0; }
.chart-card { border:1px solid var(--line); border-radius:17px; padding:14px 15px 2px; background:linear-gradient(145deg,rgba(255,255,255,.9),rgba(249,252,251,.88)); }
.insight-card { padding:17px 18px; border:1px solid rgba(54,91,99,.13); border-radius:15px; background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(247,251,249,.96)); min-height:100px; box-shadow:0 8px 22px rgba(31,66,69,.045); }
.insight-label { color:#788c95; font-size:10px; text-transform:uppercase; font-weight:800; letter-spacing:.12em; }
.insight-value { color:#193540; font:700 18px Manrope,sans-serif; margin-top:8px; }
.insight-copy { color:#738891; font-size:11px; margin-top:3px; }
.callout { border:1px solid rgba(34,141,111,.18); border-left:3px solid #249a7e; border-radius:12px; background:rgba(43,167,130,.065); color:#38525c; padding:14px 16px; font-size:13px; line-height:1.6; }
.footer { text-align:center;color:#7e9098;font-size:10px;letter-spacing:.13em;padding:26px 0 3px; }
.stButton button { border-radius:11px; min-height:43px; font-weight:750; }
.stButton button[kind="primary"] { color:#ffffff; background:linear-gradient(100deg,#168e75,#418fd4); border:0; box-shadow:0 8px 22px rgba(35,139,119,.17); }
.stDownloadButton button { border-radius:10px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.82); padding:13px 15px; border:1px solid var(--line); border-radius:13px; }
div[data-testid="stExpander"] { border-color:var(--line); border-radius:12px; background:rgba(255,255,255,.7); }
@media(max-width:760px) { .block-container{padding-top:.7rem}.hero{padding:21px 20px}.page-number{display:none}.metric-card{height:108px;padding:13px}.metric-value{font-size:18px}.brand-note{font-size:8px} }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        frame = pd.read_excel(DATA_PATH, sheet_name="Energy_Data")
    else:
        # The public deployment excludes the original household workbook.
        # Provide clearly labeled synthetic records so the analytics pages remain usable.
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=24 * 180, freq="h")
        hour = dates.hour.to_numpy()
        intensity = np.clip(3.2 + 1.8 * np.sin((hour - 7) * np.pi / 12) + rng.normal(0, 1.05, len(dates)), .2, 12)
        frame = pd.DataFrame({
            "DateTime": dates,
            "Global_reactive_power": np.clip(rng.normal(.12, .055, len(dates)), 0, .5),
            "Voltage": rng.normal(240, 4.5, len(dates)),
            "Global_intensity": intensity,
            "Sub_metering_1": rng.poisson(1.1, len(dates)),
            "Sub_metering_2": rng.poisson(1.4, len(dates)),
            "Sub_metering_3": rng.poisson(4.5, len(dates)),
        })
        frame[TARGET] = np.clip(.18 + .12 * intensity + rng.normal(0, .12, len(dates)), .05, None)
    frame["DateTime"] = pd.to_datetime(frame["DateTime"])
    frame["Hour"] = frame["DateTime"].dt.hour
    frame["DayOfWeek"] = frame["DateTime"].dt.day_name()
    return frame.sort_values("DateTime").reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def holdout_predictions(_model, frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("DateTime").reset_index(drop=True)
    split = int(len(ordered) * 0.8)
    truth = ordered.iloc[split:]
    predictions = _model.predict(prepare_features(truth))
    result = pd.DataFrame({
        "DateTime": truth["DateTime"].to_numpy(),
        "Actual (kW)": truth[TARGET].to_numpy(),
        "Predicted (kW)": predictions,
    })
    result["Residual (kW)"] = result["Actual (kW)"] - result["Predicted (kW)"]
    return result


def chart_layout(fig, height: int = 340, title: str | None = None):
    layout = dict(
        height=height,
        margin=dict(l=12, r=12, t=50 if title else 18, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#526873", size=11),
        title_font=dict(family="Manrope, sans-serif", size=14, color="#1d3843"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd9dc", font_color="#193540"),
    )
    layout["title"] = {"text": title or ""}
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="rgba(49,82,94,.10)", linecolor="rgba(49,82,94,.15)", zerolinecolor="rgba(49,82,94,.15)", tickfont=dict(color="#71838c"), title_font=dict(color="#49616b"))
    fig.update_yaxes(gridcolor="rgba(49,82,94,.10)", linecolor="rgba(49,82,94,.15)", zerolinecolor="rgba(49,82,94,.15)", tickfont=dict(color="#71838c"), title_font=dict(color="#49616b"))
    return fig


def section(title: str, subtitle: str = ""):
    st.markdown(
        f"<div class='section-head'><div><h2>{title}</h2><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def metric_card(container, label: str, value: str, note: str, icon: str):
    container.markdown(
        f"<div class='metric-card'><div class='metric-top'><span>{label}</span><span class='metric-icon'>{icon}</span></div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, description: str, number: str):
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{kicker}</div><h1>{title}</h1><p>{description}</p><div class='page-number'>{number} / 04</div></div>",
        unsafe_allow_html=True,
    )


def page_chart(fig, key: str):
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True}, key=key)
    st.markdown("</div>", unsafe_allow_html=True)


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("The trained model files are missing. Run `python train_model.py` from the project folder first.")
    st.stop()

data = load_data()
HAS_PRIVATE_DATA = DATA_PATH.exists()
model = load_model()
report = load_report()
selected = report["selected_metrics"]

with st.sidebar:
    st.markdown("<div class='brand'><span class='brand-icon'>ϟ</span><span>WattWise</span></div><div class='brand-note'>Energy intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**PROJECT SNAPSHOT**")
    st.caption("A machine learning studio for understanding and estimating household electricity demand.")
    st.caption("Historical workbook loaded." if HAS_PRIVATE_DATA else "Demo charts use synthetic sample data; the private workbook is not published.")
    st.markdown(f"**{len(data):,}** readings  ·  **{data.shape[1]}** source fields")
    st.markdown(f"{data['DateTime'].min():%b %Y} — {data['DateTime'].max():%b %Y}")
    st.markdown("---")
    st.markdown("**SELECTED MODEL**")
    st.markdown(f"<span style='color:#42e2c2;font-weight:800'>{report['selected_model']}</span>", unsafe_allow_html=True)
    st.caption(f"Time-series CV RMSE · {selected['cv_rmse']:.3f} kW")
    st.markdown("---")
    st.caption("FINAL PROJECT  ·  ENERGY CONSUMPTION")

st.markdown(
    "<div class='brandbar'><div><div class='brand'><span class='brand-icon'>ϟ</span><span>WattWise</span></div><div class='brand-note'>Household energy intelligence</div></div><div class='live-badge'>● &nbsp; MODEL READY</div></div>",
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "Website pages",
    options=["⌂  Dashboard", "▥  Explore data", "ϟ  Predict", "ⓘ  About & model"],
    default="⌂  Dashboard",
    selection_mode="single",
    label_visibility="collapsed",
    key="website_page",
)

if page is None:
    page = "⌂  Dashboard"

if page == "⌂  Dashboard":
    hero("ENERGY AT A GLANCE  ·  HOUSEHOLD METER DATA", "Understand every watt.", "Explore real meter readings, spot demand patterns, and estimate active power with a model trained on chronological data.", "01")
    metrics = st.columns(4, gap="medium")
    metric_card(metrics[0], "Measurements", f"{len(data):,}", "meter observations in the workbook", "▤")
    metric_card(metrics[1], "Typical demand", f"{data[TARGET].median():.2f} kW", "median active power reading", "◒")
    metric_card(metrics[2], "Best model", report["selected_model"], "picked by time-series validation", "✦")
    metric_card(metrics[3], "Holdout R²", f"{selected['r2']:.3f}", "latest 20% kept unseen", "↗")
    if not HAS_PRIVATE_DATA:
        st.info("This public deployment does not include the household workbook. Dashboard and explorer charts use synthetic sample records; model validation metrics below were calculated on the original chronological holdout.")

    section("Demand over time", "A smoothed view helps reveal the longer-term shape of household use.")
    daily = data.set_index("DateTime")[TARGET].resample("D").mean().dropna().reset_index()
    daily["7-day trend"] = daily[TARGET].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["DateTime"], y=daily[TARGET], mode="lines", name="Daily average", line=dict(color="rgba(125,162,255,.34)", width=1), hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} kW<extra>Daily mean</extra>"))
    fig.add_trace(go.Scatter(x=daily["DateTime"], y=daily["7-day trend"], mode="lines", name="7-day trend", line=dict(color=COLORS["teal"], width=2.5), hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} kW<extra>7-day trend</extra>"))
    fig.update_yaxes(title="Active power · kW")
    page_chart(chart_layout(fig, 370, "Daily active power"), "dashboard_daily_trend")

    section("Daily rhythm", "Compare hour of day with the weekday pattern in the readings.")
    rhythm_left, rhythm_right = st.columns([1.15, 1], gap="medium")
    with rhythm_left:
        hourly = data.groupby("Hour", as_index=False)[TARGET].mean()
        fig = px.area(hourly, x="Hour", y=TARGET)
        fig.update_traces(line_color=COLORS["teal"], fillcolor="rgba(66,226,194,.15)", hovertemplate="%{x}:00<br>%{y:.2f} kW<extra></extra>")
        fig.update_xaxes(title="Hour of day", dtick=3)
        fig.update_yaxes(title="Average active power · kW")
        page_chart(chart_layout(fig, 300, "Average demand by hour"), "dashboard_hourly")
    with rhythm_right:
        weekday = data.groupby("DayOfWeek")[TARGET].mean().reindex(WEEKDAY_ORDER).dropna().reset_index()
        fig = px.bar(weekday, x="DayOfWeek", y=TARGET, color=TARGET, color_continuous_scale=["#277a78", "#42e2c2"])
        fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:.2f} kW<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title=None, tickangle=-20)
        fig.update_yaxes(title="Average active power · kW")
        page_chart(chart_layout(fig, 300, "Average demand by weekday"), "dashboard_weekday")

    peak = data.loc[data[TARGET].idxmax()]
    insights = st.columns(3, gap="medium")
    with insights[0]:
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Highest recorded reading</div><div class='insight-value'>{peak[TARGET]:.2f} kW</div><div class='insight-copy'>{peak['DateTime']:%d %b %Y · %H:%M}</div></div>", unsafe_allow_html=True)
    with insights[1]:
        peak_hour = data.groupby("Hour")[TARGET].mean().idxmax()
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Busiest average hour</div><div class='insight-value'>{int(peak_hour):02d}:00 - {int(peak_hour + 1):02d}:00</div><div class='insight-copy'>Highest mean demand across the day</div></div>", unsafe_allow_html=True)
    with insights[2]:
        best_day = data.groupby("DayOfWeek")[TARGET].mean().idxmax()
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Highest weekday average</div><div class='insight-value'>{best_day}</div><div class='insight-copy'>Based on all available observations</div></div>", unsafe_allow_html=True)

elif page == "▥  Explore data":
    hero("DATA EXPLORER  ·  FILTER, COMPARE, DISCOVER", "Get to know the readings.", "Filter the measurement history, compare consumption patterns, and inspect how meter signals move with active power.", "02")
    control1, control2 = st.columns([1.35, 1], gap="medium")
    with control1:
        date_range = st.date_input("Measurement date range", value=(data["DateTime"].min().date(), data["DateTime"].max().date()), min_value=data["DateTime"].min().date(), max_value=data["DateTime"].max().date(), key="explore_dates")
    with control2:
        chosen_days = st.multiselect("Weekdays", WEEKDAY_ORDER, default=WEEKDAY_ORDER, key="explore_weekdays")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        filtered = data[data["DateTime"].between(start_dt, end_dt, inclusive="left")]
    else:
        filtered = data.copy()
    if chosen_days:
        filtered = filtered[filtered["DayOfWeek"].isin(chosen_days)]
    if filtered.empty:
        st.warning("No measurements match those filters. Choose a wider date range or add weekdays.")
        st.stop()

    kpis = st.columns(4, gap="medium")
    metric_card(kpis[0], "Filtered readings", f"{len(filtered):,}", "rows in current view", "▤")
    metric_card(kpis[1], "Average", f"{filtered[TARGET].mean():.2f} kW", "mean active power", "◒")
    metric_card(kpis[2], "Median", f"{filtered[TARGET].median():.2f} kW", "middle observation", "↔")
    metric_card(kpis[3], "Peak", f"{filtered[TARGET].max():.2f} kW", "highest filtered reading", "▲")

    section("When does demand rise?", "The heatmap compares hour-of-day and day-of-week averages.")
    heat = filtered.pivot_table(index="DayOfWeek", columns="Hour", values=TARGET, aggfunc="mean").reindex(WEEKDAY_ORDER)
    fig = go.Figure(go.Heatmap(z=heat.to_numpy(), x=heat.columns, y=heat.index, colorscale=[[0,"#e4f0ee"],[.45,"#79c8b2"],[1,"#126d5a"]], colorbar=dict(title="kW", tickfont=dict(color="#526873"), titlefont=dict(color="#526873")), hovertemplate="%{y}, %{x}:00<br>%{z:.2f} kW<extra></extra>"))
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title=None, autorange="reversed")
    page_chart(chart_layout(fig, 355, "Average consumption · weekday × hour"), "explore_heatmap")

    left, right = st.columns(2, gap="medium")
    with left:
        section("Consumption distribution", "See the spread of active power values in the selected period.")
        fig = px.histogram(filtered, x=TARGET, nbins=36, marginal="box", color_discrete_sequence=[COLORS["blue"]])
        fig.update_traces(hovertemplate="%{x:.2f} kW<br>%{y} readings<extra></extra>")
        fig.update_xaxes(title="Active power · kW")
        fig.update_yaxes(title="Number of readings")
        page_chart(chart_layout(fig, 320), "explore_distribution")
    with right:
        section("Signals vs. active power", "Choose a meter signal to inspect its relationship with the target.")
        signal = st.selectbox("Compare active power with", ["Global_intensity", "Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"], format_func=lambda x: x.replace("_", " "), key="explore_signal")
        scatter = filtered.sample(min(len(filtered), 2500), random_state=42) if len(filtered) > 2500 else filtered
        fig = px.scatter(scatter, x=signal, y=TARGET, color="Hour", color_continuous_scale=["#527dff", "#42e2c2"], opacity=.62)
        if scatter[signal].nunique() > 1:
            slope, intercept = np.polyfit(scatter[signal], scatter[TARGET], 1)
            trend_x = np.linspace(scatter[signal].min(), scatter[signal].max(), 60)
            fig.add_trace(go.Scatter(x=trend_x, y=slope * trend_x + intercept, mode="lines", name="Linear trend", line=dict(color=COLORS["amber"], width=2, dash="dash")))
        fig.update_layout(coloraxis_colorbar=dict(title="Hour"))
        fig.update_xaxes(title=signal.replace("_", " "))
        fig.update_yaxes(title="Active power · kW")
        page_chart(chart_layout(fig, 320), "explore_scatter")

    section("Explore the records", "Download the filtered rows or preview the source fields.")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data · CSV", csv, file_name="wattwise_filtered_readings.csv", mime="text/csv")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    st.caption(f"Showing the first {min(100, len(filtered)):,} of {len(filtered):,} matching measurements.")

elif page == "ϟ  Predict":
    hero("PREDICTION STUDIO  ·  LIVE MODEL INFERENCE", "Estimate a meter reading.", "Enter the time and electrical readings for a measurement. WattWise will estimate active power and show the model context alongside the result.", "03")
    form_col, explain_col = st.columns([1.55, 1], gap="large")
    with form_col:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("#### Measurement details")
        st.caption("Use readings from one measurement interval. Keep the units consistent with the source dataset.")
        with st.form("prediction_form", border=False):
            date_col, time_col = st.columns(2)
            input_date = date_col.date_input("Measurement date", value=datetime(2010, 1, 1).date())
            input_time = time_col.time_input("Measurement time", value=time(12, 0))
            st.markdown("**Electrical readings**")
            col1, col2, col3 = st.columns(3)
            reactive = col1.number_input("Reactive power · kW", min_value=0.0, max_value=2.0, value=0.12, step=0.01)
            voltage = col2.number_input("Voltage · V", min_value=180.0, max_value=280.0, value=240.0, step=0.1)
            intensity = col3.number_input("Intensity · A", min_value=0.0, max_value=60.0, value=4.7, step=0.1)
            col4, col5, col6 = st.columns(3)
            sub1 = col4.number_input("Kitchen sub-meter · Wh", min_value=0, max_value=100, value=1)
            sub2 = col5.number_input("Laundry sub-meter · Wh", min_value=0, max_value=100, value=1)
            sub3 = col6.number_input("Water heater / AC · Wh", min_value=0, max_value=100, value=6)
            submitted = st.form_submit_button("⚡  Estimate active power", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with explain_col:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("#### What the model uses")
        st.markdown("The trained pipeline combines your electrical readings with calendar features derived from the selected timestamp.")
        st.markdown("**Electrical signals**  \nVoltage · intensity · reactive power · three sub-meters")
        st.markdown("**Time context**  \nHour · month · day-of-year · weekday cycles")
        st.markdown("**Current model**  \n" + report["selected_model"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight-card'><div class='insight-label'>Holdout typical absolute error</div><div class='insight-value'>{selected['mae']:.3f} kW</div><div class='insight-copy'>Measured on later readings that were not used during training or model selection.</div></div>", unsafe_allow_html=True)

    if submitted:
        timestamp = pd.Timestamp(datetime.combine(input_date, input_time))
        raw = pd.DataFrame([{
            "DateTime": timestamp,
            "Global_reactive_power": reactive,
            "Voltage": voltage,
            "Global_intensity": intensity,
            "Sub_metering_1": sub1,
            "Sub_metering_2": sub2,
            "Sub_metering_3": sub3,
            "DayOfWeek": timestamp.day_name(),
        }])
        estimate = float(model.predict(prepare_features(raw))[0])
        st.session_state["last_prediction"] = {"timestamp": timestamp, "estimate": estimate, "input": raw.iloc[0].to_dict()}
        st.session_state.setdefault("prediction_history", []).append({"Measurement": timestamp, "Predicted active power (kW)": estimate})

    if st.session_state.get("last_prediction"):
        result = st.session_state["last_prediction"]
        result_left, result_right = st.columns([1, 1.5], gap="medium")
        with result_left:
            metric_card(st.container(), "Predicted active power", f"{result['estimate']:.3f} kW", f"{result['timestamp']:%A · %d %b %Y · %H:%M}", "⚡")
        with result_right:
            st.markdown(f"<div class='callout'><b>Estimate ready.</b> The {report['selected_model']} pipeline predicts <b>{result['estimate']:.3f} kW</b>. In the chronological holdout, typical absolute error was <b>{selected['mae']:.3f} kW</b>. This is a historical model error summary, not a guaranteed range for this individual estimate.</div>", unsafe_allow_html=True)

    st.markdown("---")
    section("Batch predictions", "Upload a CSV with DateTime and the six electrical readings to score several rows at once.")
    template = data[["DateTime", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].head(3).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV template", template, "wattwise_prediction_template.csv", "text/csv")
    upload = st.file_uploader("Upload prediction inputs", type=["csv"], help="Required fields: DateTime, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3.")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            required = ["DateTime", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
            missing = sorted(set(required) - set(batch.columns))
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                batch["DateTime"] = pd.to_datetime(batch["DateTime"], errors="coerce")
                batch = batch.dropna(subset=["DateTime"])
                batch["DayOfWeek"] = batch["DateTime"].dt.day_name()
                batch["Predicted_Energy_Consumption_kW"] = model.predict(prepare_features(batch))
                st.success(f"Predicted {len(batch):,} rows.")
                st.dataframe(batch, use_container_width=True, hide_index=True)
                st.download_button("Download predictions · CSV", batch.to_csv(index=False).encode("utf-8"), "wattwise_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

    history = st.session_state.get("prediction_history", [])
    if len(history) > 1:
        section("Recent estimates", "Predictions made during this app session.")
        history_df = pd.DataFrame(history)
        fig = px.line(history_df, x="Measurement", y="Predicted active power (kW)", markers=True)
        fig.update_traces(line_color=COLORS["teal"], marker_color=COLORS["blue"])
        page_chart(chart_layout(fig, 260), "prediction_history")

else:
    hero("PROJECT NOTES  ·  TRAINING, EVALUATION, DEPLOYMENT", "A model you can explain.", "See what was trained, how the holdout was protected, how the candidate models compare, and where the model’s strong score comes from.", "04")

    section("Model scorecard", "Candidate models were compared on rolling training folds; the final test period comes later in time.")
    score_cards = st.columns(4, gap="medium")
    metric_card(score_cards[0], "Selected model", report["selected_model"], "lowest training CV RMSE", "✦")
    metric_card(score_cards[1], "CV RMSE", f"{selected['cv_rmse']:.3f} kW", "mean across time-series folds", "⌁")
    metric_card(score_cards[2], "Holdout MAE", f"{selected['mae']:.3f} kW", "average absolute miss", "↔")
    metric_card(score_cards[3], "Holdout R²", f"{selected['r2']:.3f}", "variance explained on later data", "↗")

    comparison = pd.DataFrame(report["model_comparison"]).sort_values("cv_rmse")
    comp_left, comp_right = st.columns([1.05, 1], gap="medium")
    with comp_left:
        section("Candidate comparison", "Lower RMSE is better. The chosen model is highlighted.")
        formatted = comparison.rename(columns={"model":"Model", "cv_rmse":"CV RMSE · kW", "mae":"Holdout MAE · kW", "rmse":"Holdout RMSE · kW", "r2":"Holdout R²"})
        st.dataframe(formatted[["Model", "CV RMSE · kW", "Holdout MAE · kW", "Holdout RMSE · kW", "Holdout R²"]].style.format({"CV RMSE · kW":"{:.4f}", "Holdout MAE · kW":"{:.4f}", "Holdout RMSE · kW":"{:.4f}", "Holdout R²":"{:.4f}"}), use_container_width=True, hide_index=True)
        st.caption(f"Training observations: {report['train_rows']:,} · Holdout observations: {report['test_rows']:,} · Holdout starts {report['test_period_start'][:10]}.")
    with comp_right:
        section("Validation error", "Average RMSE across expanding-window training folds.")
        fig = px.bar(comparison, x="cv_rmse", y="model", orientation="h", color="model", color_discrete_sequence=[COLORS["teal"], COLORS["blue"], COLORS["violet"], COLORS["amber"]])
        fig.update_layout(showlegend=False)
        fig.update_traces(hovertemplate="%{y}<br>CV RMSE: %{x:.4f} kW<extra></extra>")
        fig.update_xaxes(title="Cross-validation RMSE · kW")
        fig.update_yaxes(title=None, autorange="reversed")
        page_chart(chart_layout(fig, 295), "model_cv_rmse")

    if HAS_PRIVATE_DATA:
        test = holdout_predictions(model, data)
        eval_left, eval_right = st.columns(2, gap="medium")
        with eval_left:
            section("Actual vs. predicted", "Points close to the diagonal represent accurate estimates.")
            fig = px.scatter(test.sample(min(len(test), 1500), random_state=7), x="Actual (kW)", y="Predicted (kW)", opacity=.55, color_discrete_sequence=[COLORS["teal"]])
            low = min(test["Actual (kW)"].min(), test["Predicted (kW)"].min())
            high = max(test["Actual (kW)"].max(), test["Predicted (kW)"].max())
            fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Ideal", line=dict(color=COLORS["amber"], dash="dash")))
            page_chart(chart_layout(fig, 315, "Chronological holdout"), "model_actual_predicted")
        with eval_right:
            section("Residual distribution", "Residual = actual minus predicted; centered near zero is desirable.")
            fig = px.histogram(test, x="Residual (kW)", nbins=40, color_discrete_sequence=[COLORS["violet"]])
            fig.add_vline(x=0, line_color=COLORS["amber"], line_dash="dash")
            fig.update_xaxes(title="Residual · kW")
            fig.update_yaxes(title="Number of readings")
            page_chart(chart_layout(fig, 315, "Holdout residuals"), "model_residuals")
    else:
        st.info("Record-level holdout charts are hidden in the public demo because they could expose the private household readings. Aggregate validation scores are shown above.")

    section("What drives the selected model?", "Largest learned coefficients from the fitted Lasso pipeline, shown by absolute size.")
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    if hasattr(estimator, "coef_"):
        names = preprocessor.get_feature_names_out()
        coef = pd.DataFrame({"Feature": names, "Coefficient": estimator.coef_})
        coef["Magnitude"] = coef["Coefficient"].abs()
        coef = coef.nlargest(12, "Magnitude").sort_values("Coefficient")
        coef["Feature"] = coef["Feature"].str.replace("numeric__", "", regex=False).str.replace("weekday__DayOfWeek_", "Weekday · ", regex=False).str.replace("_", " ")
        fig = px.bar(coef, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale=[COLORS["blue"], "#6d8194", COLORS["teal"]])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Coefficient (standardized numeric inputs)")
        fig.update_yaxes(title=None)
        page_chart(chart_layout(fig, 390, "Strongest Lasso coefficients"), "model_coefficients")

    section("How the project was built", "Four clear stages make the result reproducible and presentation-ready.")
    steps = st.columns(4, gap="medium")
    step_data = [
        ("01", "Prepare", "Sort by timestamp, derive calendar and cyclical time features, encode weekday, and impute numeric inputs in the pipeline."),
        ("02", "Train", "Tune Ridge, Lasso, Decision Tree, and Random Forest using expanding-window time-series cross-validation."),
        ("03", "Evaluate", "Choose by training CV RMSE, then report MAE, RMSE, and R² once on the newest 20% holdout."),
        ("04", "Deploy", "Load the fitted preprocessing-plus-model pipeline in Streamlit for single and batch predictions."),
    ]
    for col, (number, title, copy) in zip(steps, step_data):
        with col:
            st.markdown(f"<div class='insight-card'><div class='insight-label'>STEP {number}</div><div class='insight-value'>{title}</div><div class='insight-copy'>{copy}</div></div>", unsafe_allow_html=True)

    section("A note on interpreting this score", "Use this caveat in your project presentation.")
    st.markdown("<div class='callout'>The target is closely related to the electrical meter features supplied to the model. The very high holdout R² means those readings estimate active power well in this dataset; it does not show that the app can forecast future demand without current meter readings. Weather, occupancy, appliance schedules, and external household factors are not in this workbook.</div>", unsafe_allow_html=True)
    with st.expander("Project files and commands"):
        st.code("app.py                      # Four-page Streamlit website\ntrain_model.py              # feature engineering, model search, evaluation\nartifacts/energy_model.joblib\nartifacts/metrics.json\nenergy_consumption_real_10000.xlsx\nrequirements.txt\n\npython train_model.py\nstreamlit run app.py", language="text")

st.markdown("<div class='footer'>WATTWISE &nbsp; · &nbsp; ENERGY CONSUMPTION PREDICTION &nbsp; · &nbsp; BUILT WITH STREAMLIT & SCIKIT-LEARN</div>", unsafe_allow_html=True)
