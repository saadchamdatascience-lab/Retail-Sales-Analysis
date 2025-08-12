# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from datetime import datetime

# Page config
st.set_page_config(page_title="Retail Insights Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---- CSS / Styling ----
st.markdown(
    """
    <style>
    .header {
      display:flex; align-items:center; gap:12px;
    }
    .brand {
      font-weight:700; font-size:20px;
    }
    .kpi {
      background: linear-gradient(90deg, #ffffff, #f8fafc);
      padding:14px; border-radius:10px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .small {
      color: #6b7280; font-size:12px;
    }
    .big {
      font-weight:700; font-size:20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---- Header / Hero ----
with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="header"><div class="brand">📈 Retail Insights Dashboard</div></div>', unsafe_allow_html=True)
        st.markdown("**Upload your sales CSV/Excel** to view interactive visualizations and demand forecasts.")
    with col2:
        st.image("https://static.streamlit.io/examples/dice.jpg", width=80)  # placeholder logo

st.markdown("---")

# ---- Sidebar: Upload & Controls ----
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use sample data instead", value=False)
forecast_enable = st.sidebar.checkbox("Enable Forecasting (Prophet)", value=False)
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30)

# ---- Data Loading Helpers ----
@st.cache_data
def read_file(file):
    try:
        if hasattr(file, "read"):
            raw = file.read()
            try:
                return pd.read_csv(io.BytesIO(raw))
            except Exception:
                return pd.read_excel(io.BytesIO(raw))
        else:
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        return pd.DataFrame()

def safe_dt(series):
    return pd.to_datetime(series, errors="coerce")

# ---- Provide a realistic sample dataset ----
def sample_dataset():
    rng = pd.date_range(end=pd.Timestamp.today(), periods=180)
    products = ["Beauty", "Clothing", "Electronics", "Grocery", "Home"]
    rows = []
    for d in rng:
        for p in np.random.choice(products, 3, replace=False):
            qty = max(1, int(np.random.poisson(3)))
            unit_price = int(np.random.uniform(40, 1200))
            amt = qty * unit_price
            rows.append({
                "Transaction ID": np.random.randint(100000,999999),
                "Date": d.strftime("%Y-%m-%d"),
                "Customer ID": f"CUST{np.random.randint(1000,9999)}",
                "Gender": np.random.choice(["Male","Female"]),
                "Age": np.random.randint(18,65),
                "Product Category": p,
                "Quantity": qty,
                "Price per Unit": unit_price,
                "Total Amount": amt
            })
    return pd.DataFrame(rows)

# ---- Load data (upload / sample / repo fallback) ----
if uploaded_file and not use_sample:
    df = read_file(uploaded_file)
elif use_sample:
    df = sample_dataset()
else:
    # try to use a repo dataset if present
    try:
        df = read_file("retail_sales_dataset.csv")
        if df.empty:
            st.info("No file uploaded and no sample selected — enable 'Use sample data' or upload a file.")
    except Exception:
        df = pd.DataFrame()
        st.info("No file uploaded and no sample selected — enable 'Use sample data' or upload a file.")

if df is None or df.empty:
    st.stop()

# ---- Normalize columns (strip) ----
df.columns = [c.strip() for c in df.columns]

# ---- Detect date & sales column ----
date_col = None
for c in df.columns:
    if "date" in c.lower():
        date_col = c
        break

if date_col:
    df[date_col] = safe_dt(df[date_col])

# Determine sales column
sales_candidates = [c for c in df.columns if any(k in c.lower() for k in ["total", "amount", "sales"])]
sales_col = sales_candidates[0] if sales_candidates else None

if not sales_col:
    # try compute from quantity * price
    qty_col = next((c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()), None)
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    if qty_col and price_col:
        df["Total Amount"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0) * pd.to_numeric(df[price_col], errors="coerce").fillna(0)
        sales_col = "Total Amount"

if not date_col:
    st.warning("No Date column detected. Time-based charts and forecasting require a Date column.")

if not sales_col:
    st.warning("No sales column detected. Add a Total/Amount/Sales column or include Quantity and Price per Unit.")
    
# ---- Basic cleans ----
initial_rows = len(df)
df = df.drop_duplicates().reset_index(drop=True)
dropped = initial_rows - len(df)

# ---- KPIs ----
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    total_sales = int(df[sales_col].sum()) if sales_col in df.columns else 0
    total_transactions = len(df)
    avg_order = round(df[sales_col].mean(),2) if sales_col in df.columns else 0
    unique_products = df["Product Category"].nunique() if "Product Category" in df.columns else "-"
    k1.metric("Total Sales", f"₹{total_sales:,}")
    k2.metric("Transactions", f"{total_transactions}")
    k3.metric("Avg Order Value", f"₹{avg_order:,}")
    k4.metric("Product Categories", f"{unique_products}")

st.markdown("---")

# ---- Data preview & summary ----
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Summary Statistics")
try:
    st.write(df.describe(include="all"))
except Exception:
    st.write("Could not compute full summary for non-numeric columns.")

# ---- Visualizations: layout two columns ----
st.subheader("Visualizations")
left, right = st.columns((2,1))

with left:
    if date_col and sales_col in df.columns:
        ts = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)
        fig_ts = px.line(ts, x=date_col, y=sales_col, title="Total Sales Over Time", markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Time-series requires Date + Sales.")

    # top products chart
    if "Product Category" in df.columns and sales_col in df.columns:
        prod = df.groupby("Product Category")[sales_col].sum().reset_index().sort_values(sales_col, ascending=False).head(12)
        fig_prod = px.bar(prod, x="Product Category", y=sales_col, title="Top Product Categories by Sales")
        st.plotly_chart(fig_prod, use_container_width=True)

with right:
    if "Gender" in df.columns and sales_col in df.columns:
        g = df.groupby("Gender")[sales_col].sum().reset_index()
        fig_g = px.pie(g, names="Gender", values=sales_col, title="Sales by Gender")
        st.plotly_chart(fig_g, use_container_width=True)
    if "Age" in df.columns and sales_col in df.columns:
        bins = [0,18,25,35,50,100]
        labels = ["<18","18-25","26-35","36-50","50+"]
        df["Age Group"] = pd.cut(pd.to_numeric(df["Age"], errors="coerce").fillna(0), bins=bins, labels=labels, right=False)
        age = df.groupby("Age Group")[sales_col].sum().reset_index()
        fig_age = px.bar(age, x="Age Group", y=sales_col, title="Sales by Age Group")
        st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")

# ---- Download cleaned CSV ----
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button("Download cleaned data (CSV)", to_csv_bytes(df), "cleaned_retail_data.csv", "text/csv")

# ---- Forecasting (optional) ----
if forecast_enable:
    st.subheader("Demand Forecasting")
    if date_col and sales_col in df.columns:
        try:
            from prophet import Prophet
            df_ts = df.groupby(date_col)[sales_col].sum().reset_index().rename(columns={date_col: "ds", sales_col: "y"})
            df_ts = df_ts.dropna()
            # run forecast on button
            if st.button("Run forecast"):
                m = Prophet()
                m.fit(df_ts)
                future = m.make_future_dataframe(periods=int(forecast_days))
                fc = m.predict(future)
                figf = px.line(fc, x="ds", y="yhat", title="Forecast (yhat)")
                figf.add_scatter(x=df_ts["ds"], y=df_ts["y"], mode="lines", name="Actual")
                st.plotly_chart(figf, use_container_width=True)
                st.dataframe(fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(30))
                st.download_button("Download forecast CSV", fc[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")
        except Exception as e:
            st.warning("Forecasting library not available. Add 'prophet' to requirements.txt and redeploy. Error: " + str(e))
    else:
        st.info("Forecasting requires a Date column and a Sales column.")

st.markdown("---")
st.caption("Built with Streamlit — customize styling or give me the live site URL for further visual matching.")
