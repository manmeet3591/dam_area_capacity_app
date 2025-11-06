# app.py
import os
import json
import math
import datetime as dt

import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium

import ee

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Panshet Dam - Monthly Water Area (Dynamic World)",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# LOGIN GATE (fixed credentials for demo)
# --------------------------------------------------------------------------------------
VALID_USER = "ndsa"
VALID_PASS = "cwprs"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_form():
    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if u == VALID_USER and p == VALID_PASS:
            st.session_state.authenticated = True
            st.success("Logged in!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.authenticated:
    login_form()
    st.stop()

# --------------------------------------------------------------------------------------
# Earth Engine initialization
# --------------------------------------------------------------------------------------
def init_ee():
    """
    Initialize Google Earth Engine using a service account key provided via:
    - Streamlit secrets: st.secrets["GEE_SERVICE_ACCOUNT_KEY"] = {client_email, private_key}
    - OR environment variable: GEE_SERVICE_ACCOUNT_KEY (same JSON string)
    """
    key_info = None

    # Prefer Streamlit secrets
    if "GEE_SERVICE_ACCOUNT_KEY" in st.secrets:
        key_info = st.secrets["GEE_SERVICE_ACCOUNT_KEY"]
    else:
        # Fallback: env var with JSON string
        env_key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "")
        if env_key:
            try:
                key_info = json.loads(env_key)
            except Exception:
                st.error("Failed to parse GEE_SERVICE_ACCOUNT_KEY from environment.")
                st.stop()

    if not key_info or "client_email" not in key_info or "private_key" not in key_info:
        st.error(
            "Missing GEE service account credentials.\n\n"
            "Add GEE_SERVICE_ACCOUNT_KEY with 'client_email' and 'private_key' "
            "to Streamlit secrets or set it as an environment variable."
        )
        st.stop()

    try:
        credentials = ee.ServiceAccountCredentials(
            email=key_info["client_email"],
            key_data=key_info["private_key"]
        )
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        st.stop()

init_ee()

# --------------------------------------------------------------------------------------
# Constants / Defaults
# --------------------------------------------------------------------------------------
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DEFAULT_START = dt.date(2024, 6, 1)   # June 1, 2024
DEFAULT_END   = dt.date(2025, 6, 30)  # June 30, 2025 (inclusive intent; handled below)

# Approx center of Panshet (Tanaji Sagar) Dam, Pune, India.
DEFAULT_CENTER_LAT = 18.405
DEFAULT_CENTER_LON = 73.585

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def km_to_deg_lat(km: float) -> float:
    return km / 111.32  # ~111.32 km per degree latitude

def km_to_deg_lon(km: float, at_lat_deg: float) -> float:
    # prevent division by zero at poles with tiny epsilon
    return km / (111.32 * max(math.cos(math.radians(at_lat_deg)), 1e-6))

def make_square_box(center_lat: float, center_lon: float, side_km: float):
    """
    Build a square AOI (side_km × side_km) around a lat/lon center.
    Returns (ee.Geometry, (min_lon, min_lat, max_lon, max_lat))
    """
    half_km = side_km / 2.0
    dlat = km_to_deg_lat(half_km)
    dlon = km_to_deg_lon(half_km, center_lat)
    min_lon = center_lon - dlon
    max_lon = center_lon + dlon
    min_lat = center_lat - dlat
    max_lat = center_lat + dlat
    geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    return geom, (min_lon, min_lat, max_lon, max_lat)

def month_ranges(start_date: dt.date, end_date: dt.date):
    """
    Yield (month_start_date, month_end_date_exclusive) pairs that cover [start_date, end_date].
    """
    # start at first day of start_date's month
    cur = dt.date(start_date.year, start_date.month, 1)

    # end_excl is first day of month after end_date
    if end_date.month == 12:
        end_excl = dt.date(end_date.year + 1, 1, 1)
    else:
        end_excl = dt.date(end_date.year, end_date.month + 1, 1)

    while cur < end_excl:
        # next month
        if cur.month == 12:
            nxt = dt.date(cur.year + 1, 1, 1)
        else:
            nxt = dt.date(cur.year, cur.month + 1, 1)

        # clip subrange to requested interval
        start = max(cur, start_date)
        end = min(nxt, end_excl)
        if start < end:
            yield start, end

        cur = nxt

def compute_monthly_water_area(aoi: ee.Geometry, start_date: dt.date, end_date: dt.date, threshold: float = 0.5):
    """
    For each month in [start_date, end_date], compute monthly mean of DW 'water' band,
    threshold it (default 0.5), count 10 m pixels in AOI, area = pixel_count * 100 m².

    Returns a pandas DataFrame: ['month', 'pixel_count', 'area_m2']
    """
    rows = []

    for m_start, m_end in month_ranges(start_date, end_date):
        col = (
            ee.ImageCollection(DW_COLLECTION)
            .filterBounds(aoi)
            .filterDate(ee.Date(str(m_start)), ee.Date(str(m_end)))
        )

        # Skip empty months cleanly
        if col.size().getInfo() == 0:
            rows.append({
                "month": m_start.strftime("%Y-%m"),
                "pixel_count": 0,
                "area_m2": 0.0
            })
            continue

        monthly_mean = col.select("water").mean()
        binary = monthly_mean.gt(threshold)

        # Count pixels by summing an image of ones masked by 'binary'
        ones = ee.Image.constant(1).updateMask(binary)
        pixel_count = ones.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,           # Dynamic World nominal scale
            maxPixels=1e12,
            bestEffort=True
        ).get("constant")

        try:
            pixel_count = int(pixel_count.getInfo()) if pixel_count is not None else 0
        except Exception:
            pixel_count = 0

        area_m2 = pixel_count * 100.0  # 10 m × 10 m → 100 m² per pixel

        rows.append({
            "month": m_start.strftime("%Y-%m"),
            "pixel_count": pixel_count,
            "area_m2": area_m2
        })

    return pd.DataFrame(rows)

def make_map(aoi_bounds, center_lat, center_lon, zoom=12):
    """
    Build a folium map showing the AOI rectangle.
    """
    (min_lon, min_lat, max_lon, max_lat) = aoi_bounds
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=True)
    # AOI rectangle
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        fill=True, fill_opacity=0.1,
        color="blue", weight=2,
    ).add_to(m)
    folium.Marker([center_lat, center_lon], tooltip="AOI center").add_to(m)
    return m

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("Dynamic World • Monthly Water Area — Panshet Dam (2 km × 2 km)")

colA, colB = st.columns([1, 1])
with colA:
    start_date = st.date_input("Start date", value=DEFAULT_START)
with colB:
    end_date = st.date_input("End date (inclusive)", value=DEFAULT_END)

if start_date > end_date:
    st.error("Start date must be on or before End date.")
    st.stop()

st.markdown("**Area of Interest (AOI):** Square centered on Panshet Dam, Pune. Default **2 km × 2 km** (adjustable).")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    center_lat = st.number_input("Center latitude", value=DEFAULT_CENTER_LAT, format="%.6f")
with c2:
    center_lon = st.number_input("Center longitude", value=DEFAULT_CENTER_LON, format="%.6f")
with c3:
    side_km = st.number_input("Square side length (km)", value=2.0, min_value=0.5, step=0.5, format="%.1f")

threshold = st.slider(
    "Water probability threshold (applied to monthly mean of Dynamic World 'water' band)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

# --------------------------------------------------------------------------------------
# AOI & Map
# --------------------------------------------------------------------------------------
aoi, aoi_bounds = make_square_box(center_lat, center_lon, side_km)

st.subheader("AOI Map")
m = make_map(aoi_bounds, center_lat, center_lon, zoom=12)
st_folium(m, width=900, height=500)

# --------------------------------------------------------------------------------------
# Compute Monthly Results
# --------------------------------------------------------------------------------------
with st.spinner("Computing monthly water area from Dynamic World..."):
    df = compute_monthly_water_area(aoi, start_date, end_date, threshold=threshold)

if df.empty:
    st.warning("No results returned for the selected range.")
    st.stop()

# Prepare for display
df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
df = df.sort_values("month_dt")
df["area_km2"] = df["area_m2"] / 1e6

# --------------------------------------------------------------------------------------
# Results Table & Bar Plot
# --------------------------------------------------------------------------------------
st.subheader("Monthly Water Area")
st.dataframe(
    df[["month", "pixel_count", "area_m2", "area_km2"]].rename(columns={
        "month": "Month",
        "pixel_count": "Pixel count (> threshold)",
        "area_m2": "Water area (m²)",
        "area_km2": "Water area (km²)",
    }),
    use_container_width=True,
)

st.subheader("Bar Chart: Monthly Water Area (km²)")
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("month:N", title="Month", sort=None),
        y=alt.Y("area_km2:Q", title="Water area (km²)"),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("area_km2:Q", title="Area (km²)", format=".4f"),
            alt.Tooltip("area_m2:Q", title="Area (m²)", format=",.0f"),
            alt.Tooltip("pixel_count:Q", title="Pixels"),
        ],
    )
)
st.altair_chart(chart, use_container_width=True)

# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
with st.expander("Method & Notes"):
    st.markdown(
        """
- **Dataset:** `GOOGLE/DYNAMICWORLD/V1` at **10 m** resolution; we use the **`water`** probability band.
- **Monthly aggregation:** For each month, we take the **mean** water probability over all images and then
  **threshold** it (default **0.5**).
- **Area:** Count of 10 m pixels above threshold × **100 m²** per pixel → **m²**, and divide by **1e6** for **km²**.
- **AOI:** Square you select (default **2 km × 2 km**) centered on the provided coordinates.
- **Tip:** For monsoon-season dynamics, explore thresholds ~**0.4–0.6**.
        """
    )
