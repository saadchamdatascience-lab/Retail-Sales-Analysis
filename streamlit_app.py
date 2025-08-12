import streamlit as st
import pandas as pd

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("📊 First 5 rows of your file:")
    st.write(df.head())

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Sales Amount'] = pd.to_numeric(df['Sales Amount'], errors='coerce')

st.write(f"✅ Total Rows After Cleaning: {len(df)}")
st.write(f"✅ Date Range: {df['Date'].min()} to {df['Date'].max()}")


