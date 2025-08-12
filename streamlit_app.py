# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(page_title="Retail Sales Analysis", layout="wide")
st.title("📊 Retail Sales Analysis & Forecasting")

# ----------------------
# Helpers
# ----------------------
@st.cache_data
def read_file(file) -> pd.DataFrame:
    try:
        if hasattr(file, "read"):
            # file-like (uploaded)
            content = file.read()
            # attempt csv then excel
            try:
                return pd.read_csv(io.BytesIO(content))
            except Exception:
                return pd.read_excel(io.BytesIO(content))
        else:
            # path string
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        return pd.DataFrame()

def detect_sales_column(df: pd.DataFrame):
    """Return column name to use as sales. Tries common names, else computes Quantity*Price."""
    candidates = ["Total Amount", "Total", "Sales", "Sales Amount", "Amount", "total_amount", "sales"]
    for c in candidates:
        if c in df.columns:
            return c
    # try compute from Quantity * Price
    qty_cols = [c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()]
    price_cols = [c for c in df.columns if "price" in c.lower() or "unit price" in c.lower() or "unit_price" in c.lower()]
    if qty_cols and price_cols:
        return (qty_cols[0], price_cols[0])  # return tuple to indicate multiplication
    return None

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

# ----------------------
# UI - Sidebar options
# ----------------------
st.sidebar.header("Upload / Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use sample demo data (no upload)", value=False)
forecast_enable = st.sidebar.checkbox("Enable forecasting (Prophet)", value=False)

# ----------------------
# Load data (uploaded / sample / repo file)
# ----------------------
if uploaded_file and not use_sample:
    df = read_file(uploaded_file)
elif use_sample:
    # Build a realistic sample dataset similar to your earlier CSV columns
    rng = pd.date_range(end=pd.Timestamp.today(), periods=180)
    products = ["Beauty", "Clothing", "Electronics", "Grocery", "Home"]
    data = []
    for d in rng:
        for p in np.random.choice(products, size=3, replace=False):
            qty = int(np.random.poisson(3) + 1)
            unit_price = int(np.random.uniform(50, 2000))
            amt = qty * unit_price
            data.append([d.strftime("%Y-%m-%d"), f"CUST{np.random.randint(1000,9999)}",
                         np.random.choice(["Male", "Female"]), np.random.randint(18,70),
                         p, qty, unit_price, amt])
    df = pd.DataFrame(data, columns=[
        "Date", "Customer ID", "Gender", "Age", "Product Category",
        "Quantity", "Price per Unit", "Total Amount"
    ])
else:
    # Try to read a repo file if present (helpful for testing)
    try:
        df = read_file("retail_sales_dataset.csv")
        if df.empty:
            st.info("No uploaded file and no sample selected. Upload a file or enable sample data.")
    except Exception:
        df = pd.DataFrame()
        st.info("No uploaded file and no sample selected. Upload a file or enable sample data.")

if df is None or df.empty:
    st.stop()

# ----------------------
# Preprocess & cleaning
# ----------------------
st.subheader("1) Data preview & cleaning")

# show raw preview
with st.expander("Raw data (first rows)"):
    st.dataframe(df.head(10))

# Standardize column names (strip)
df.columns = [c.strip() for c in df.columns]

# Detect date column
date_col_candidates = [c for c in df.columns if "date" in c.lower()]
if date_col_candidates:
    date_col = date_col_candidates[0]
    df[date_col] = safe_to_datetime(df[date_col])
else:
    date_col = None

if not date_col:
    st.warning("No date column detected. Forecasts and time charts require a date column named like 'Date'.")
else:
    st.write(f"Detected date column: **{date_col}**")
    st.write(f"Date range (after parse): {df[date_col].min()} — {df[date_col].max()}")

# Detect sales column
sales_col = detect_sales_column(df)
if isinstance(sales_col, tuple):
    qty_col, price_col = sales_col
    st.write(f"Computed sales as **{qty_col} * {price_col}**")
    df["Sales"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0) * pd.to_numeric(df[price_col], errors="coerce").fillna(0)
    sales_col = "Sales"
elif isinstance(sales_col, str):
    st.write(f"Detected sales column: **{sales_col}**")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
else:
    st.warning("No sales column detected. Please ensure your file has 'Total Amount' or similar, or includes Quantity and Price per Unit.")
    # continue, but many charts will be unavailable

# Basic cleaning actions
initial_rows = len(df)
df = df.drop_duplicates()
removed = initial_rows - len(df)
st.write(f"Removed {removed} duplicate rows.")

# Show missing values
nulls = df.isnull().sum()
with st.expander("Column null counts"):
    st.write(nulls[nulls > 0])

# Option to drop rows with missing critical values
if st.button("Drop rows with missing Date or Sales"):
    if date_col:
        df = df.dropna(subset=[date_col])
    if sales_col:
        df = df.dropna(subset=[sales_col])
    st.success("Dropped rows with missing date/sales where present.")
    st.experimental_rerun()

# ----------------------
# Display basic stats
# ----------------------
st.subheader("2) Summary statistics")
try:
    st.write(df.describe(include="all"))
except Exception as e:
    st.write("Could not compute full describe:", e)

# ----------------------
# Visualizations
# ----------------------
st.subheader("3) Visualizations")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("**Sales over time**")
    if date_col and sales_col:
        # aggregate daily
        timeseries = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)
        fig = px.line(timeseries, x=date_col, y=sales_col, title="Total Sales Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Time series requires a Date column and Sales column.")

    st.markdown("**Top products (by sales)**")
    prod_col_candidates = [c for c in df.columns if "product" in c.lower() or "product category" in c.lower() or "product" in c.lower()]
    if prod_col_candidates and sales_col:
        prod_col = prod_col_candidates[0]
        prod_sales = df.groupby(prod_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False).head(10)
        fig2 = px.bar(prod_sales, x=prod_col, y=sales_col, title="Top Products by Sales")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No product column detected or no sales column.")

with col2:
    st.markdown("**Sales by Gender**")
    if "Gender" in df.columns and sales_col:
        g = df.groupby("Gender")[sales_col].sum().reset_index()
        figg = px.pie(g, names="Gender", values=sales_col, title="Sales by Gender")
        st.plotly_chart(figg, use_container_width=True)
    else:
        st.info("No Gender column found or Sales column missing.")

    st.markdown("**Sales by Age Group**")
    if "Age" in df.columns and sales_col:
        bins = [0,18,25,35,50,100]
        labels = ["<18","18-25","26-35","36-50","50+"]
        try:
            df["Age Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce").fillna(0), bins=bins, labels=labels, right=False)
            ag = df.groupby("Age Group")[sales_col].sum().reset_index()
            fage = px.bar(ag, x="Age Group", y=sales_col, title="Sales by Age Group")
            st.plotly_chart(fage, use_container_width=True)
        except Exception as e:
            st.info("Could not compute Age Group chart: " + str(e))
    else:
        st.info("No Age column found or Sales column missing.")

# ----------------------
# Download cleaned dataset
# ----------------------
st.subheader("4) Download cleaned data")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_bytes = convert_df_to_csv(df)
st.download_button("Download cleaned CSV", csv_bytes, "cleaned_retail_data.csv", "text/csv")

# ----------------------
# Forecasting (optional)
# ----------------------
if forecast_enable:
    st.subheader("5) Forecasting (Prophet)")
    try:
        from prophet import Prophet
        # prepare series
        if date_col and sales_col:
            df_ts = df.groupby(date_col)[sales_col].sum().reset_index().rename(columns={date_col: "ds", sales_col: "y"})
            df_ts = df_ts.dropna()
            periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30)
            if st.button("Run forecast"):
                m = Prophet()
                m.fit(df_ts)
                future = m.make_future_dataframe(periods=int(periods))
                forecast = m.predict(future)
                # plot
                figf = px.line(forecast, x="ds", y="yhat", title="Forecast (yhat)")
                figf.add_scatter(x=df_ts["ds"], y=df_ts["y"], mode="lines", name="Actual")
                st.plotly_chart(figf, use_container_width=True)
                # show forecast table
                with st.expander("Forecast table (next rows)"):
                    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods))
                # download forecast
                st.download_button("Download forecast CSV", forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False).encode('utf-8'), "forecast.csv", "text/csv")
        else:
            st.info("Forecasting requires a Date column and a Sales column.")
    except Exception as e:
        st.warning("Prophet not available or an error occurred. To enable forecasting, add `prophet` to requirements.txt and redeploy. Error: " + str(e))

st.markdown("---")
st.markdown("Made with ❤️ — upload your CSV or use the sample to begin.")
