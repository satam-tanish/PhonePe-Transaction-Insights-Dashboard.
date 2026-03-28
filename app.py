import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="PhonePe Dashboard", layout="wide")

# -------------------------
# PREMIUM UI STYLE
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 PhonePe Transaction Insights Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
file_path = "data/transaction_data.csv"

if not os.path.exists(file_path):
    st.error("❌ transaction_data.csv not found")
    st.stop()

df = pd.read_csv(file_path)

if df.empty:
    st.error("❌ Dataset is empty")
    st.stop()

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔍 Filters")

state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
quarter = st.sidebar.selectbox("Quarter", sorted(df["Quarter"].unique()))

filtered_df = df[
    (df["State"] == state) &
    (df["Year"] == year) &
    (df["Quarter"] == quarter)
]

if filtered_df.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# -------------------------
# METRICS
# -------------------------
total_amount = filtered_df["Amount"].sum()
total_count = filtered_df["Count"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Amount", f"₹ {total_amount:,.0f}")
col2.metric("🔢 Transactions", f"{total_count:,}")
col3.metric("📍 States Covered", df["State"].nunique())

st.divider()

# -------------------------
# BAR CHART
# -------------------------
st.subheader("💳 Transaction Breakdown")

chart_data = filtered_df.groupby("Transaction_Type")["Amount"].sum()
st.bar_chart(chart_data)

# -------------------------
# INDIA MAP (SAFE VERSION)
# -------------------------
st.subheader("🌍 India State-wise Transactions")

map_df = df.groupby("State")["Amount"].sum().reset_index()

fig = px.choropleth(
    map_df,
    locations="State",
    locationmode="country names",
    color="Amount",
    color_continuous_scale="Reds",
    title="Total Transactions by State"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TREND GRAPH
# -------------------------
st.subheader("📈 Transaction Trend")

trend = df.groupby("Year")["Amount"].sum()
st.line_chart(trend)

# -------------------------
# AI PREDICTION
# -------------------------
st.subheader("🤖 AI Prediction")

model_df = df.groupby("Year")["Amount"].sum().reset_index()

X = model_df["Year"].values.reshape(-1, 1)
y = model_df["Amount"].values

model = LinearRegression()
model.fit(X, y)

next_year = max(model_df["Year"]) + 1
prediction = model.predict([[next_year]])

st.success(f"📊 Predicted Transaction for {next_year}: ₹ {int(prediction[0]):,}")

# -------------------------
# TOP STATES
# -------------------------
st.subheader("🏆 Top 10 States")

top_states = (
    df.groupby("State")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_states)

# -------------------------
# RAW DATA
# -------------------------
st.subheader("📄 Raw Data")

if st.checkbox("Show Data"):
    st.dataframe(df)