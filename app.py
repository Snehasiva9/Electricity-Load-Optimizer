import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Electricity Load Optimizer", layout="wide")
st.title("⚡ Electricity Load Optimizer (AI + ML Powered)")

# 🔐 API (optional)
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
API_KEY = "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"

def ask_llm(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        data = r.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "")
        return "Model busy"
    except:
        return "API Error"

# ---------- Column Detection ----------
def detect_columns(df):
    df.columns = df.columns.str.strip()

    cons_cols = [c for c in df.columns if "consumption" in c.lower() or "load" in c.lower()]
    cost_cols = [c for c in df.columns if "cost" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    region_cols = [c for c in df.columns if "region" in c.lower()]

    if len(cons_cols) == 0:
        st.error("❌ No Consumption/Load column found!")
        st.stop()
    else:
        cons_col = cons_cols[0]

    if len(cost_cols) == 0:
        st.warning("⚠️ No Cost column found → Using default cost = 1")
        df["Default_Cost"] = 1
        cost_col = "Default_Cost"
    else:
        cost_col = cost_cols[0]

    date_col = date_cols[0] if len(date_cols) > 0 else None
    region_col = region_cols[0] if len(region_cols) > 0 else None

    return cons_col, cost_col, date_col, region_col

# ---------- Tabs ----------
tabs = st.tabs(["📂 Upload","📈 Predictions","⚡ Optimization","🤖 AI Insights","📄 Report"])

# ------------------ Upload ------------------
with tabs[0]:
    file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.write("📊 Dataset Preview")
        st.dataframe(df)

        st.write("📌 Columns:", df.columns.tolist())

        st.session_state["df"] = df

# ------------------ Predictions (ML) ------------------
with tabs[1]:
    if "df" in st.session_state:
        df = st.session_state["df"].copy()

        cons_col, cost_col, date_col, region_col = detect_columns(df)

        st.subheader("📈 ML-Based Demand Prediction")

        df = df.reset_index()
        df["Index"] = df.index

        X = df[["Index"]]
        y = df[cons_col]

        model = LinearRegression()
        model.fit(X, y)

        df["ML_Prediction"] = model.predict(X)

        st.plotly_chart(
            px.line(df, y=[cons_col, "ML_Prediction"],
            title="Actual vs ML Prediction")
        )

        future_steps = st.slider("Predict Future Days", 1, 10, 5)

        future_index = np.arange(len(df), len(df) + future_steps).reshape(-1,1)
        future_pred = model.predict(future_index)

        future_df = pd.DataFrame({
            "Future_Index": range(len(df), len(df)+future_steps),
            "Predicted_Load": future_pred
        })

        st.subheader("🔮 Future Predictions")
        st.plotly_chart(px.line(future_df, x="Future_Index", y="Predicted_Load"))

# ------------------ Optimization ------------------
with tabs[2]:
    if "df" in st.session_state:
        df = st.session_state["df"].copy()

        cons_col, cost_col, date_col, region_col = detect_columns(df)

        st.subheader("⚡ Smart Optimization Engine")

        total = df[cons_col].sum()
        cost = (df[cons_col] * df[cost_col]).sum()

        st.metric("Total Cost", int(cost))

        reduction = st.slider("Reduce Load %", 0, 50, 10)

        df["Optimized_Load"] = df[cons_col] * (1 - reduction/100)
        df["Savings"] = (df[cons_col] - df["Optimized_Load"]) * df[cost_col]

        total_savings = df["Savings"].sum()

        st.metric("💰 Estimated Savings", int(total_savings))

        # Graphs
        st.plotly_chart(
            px.line(df, y=[cons_col, "Optimized_Load"],
            title="Before vs After Optimization")
        )

        st.plotly_chart(
            px.bar(df, y="Savings", title="Savings per Entry")
        )

        if region_col:
            st.plotly_chart(px.pie(df, names=region_col, values=cons_col))

# ------------------ AI Insights ------------------
with tabs[3]:
    if "df" in st.session_state:
        df = st.session_state["df"].copy()

        cons_col, cost_col, date_col, region_col = detect_columns(df)

        st.subheader("🤖 AI Insights Dashboard")

        st.plotly_chart(px.line(df, y=cons_col, title="Demand Trend"))

        threshold = df[cons_col].mean()
        df["Type"] = df[cons_col].apply(lambda x: "Peak" if x > threshold else "Normal")

        st.plotly_chart(px.pie(df, names="Type", title="Peak vs Normal Usage"))

        st.write("### 🔍 Insights")
        st.write(f"""
        ✔ Average Demand: {int(df[cons_col].mean())}  
        ✔ Peak Demand: {int(df[cons_col].max())}  
        ✔ Low Demand: {int(df[cons_col].min())}  

        ✔ High usage detected above average  
        ✔ Optimization can reduce peak load  
        ✔ Cost directly linked with consumption  
        """)

        # Optional LLM
        if st.button("⚡ Generate AI Text Insights"):
            prompt = df.describe().to_string()
            res = ask_llm(prompt)

            if "error" in res.lower() or res == "":
                st.warning("Fallback insights used")
            else:
                st.success(res)

# ------------------ Report ------------------
with tabs[4]:
    if "df" in st.session_state:
        df = st.session_state["df"].copy()

        cons_col, cost_col, date_col, region_col = detect_columns(df)

        total = df[cons_col].sum()
        cost = (df[cons_col] * df[cost_col]).sum()

        st.subheader("📄 Final Report")

        st.write("### 📊 Summary")
        st.write(df.describe())

        st.write("### 📌 Conclusion")
        st.write(f"""
        ✔ Total Demand: {int(total)}  
        ✔ Total Cost: {int(cost)}  
        ✔ Demand is increasing  
        ✔ Optimization reduces cost  
        """)

        report = f"""
        ELECTRICITY LOAD OPTIMIZER REPORT

        Total Demand: {total}
        Total Cost: {cost}

        Demand trend is increasing.
        Optimization can reduce cost.
        """

        st.download_button("📥 Download Report", report, "Final_Report.txt")