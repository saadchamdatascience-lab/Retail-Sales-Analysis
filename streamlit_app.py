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

