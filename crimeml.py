
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import json
from plotly.subplots import make_subplots

df2 = pd.read_pickle("df2.pkl")
df_model = joblib.load("df_model.pkl")
lightgbm_model = joblib.load("lightgbm_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
df_sample = pd.read_pickle("df_sample.pkl")



st.sidebar.title("Navigation")
st.set_page_config(
    page_title="Chicago Crime ML Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Set formatting for Page words
st.markdown("""
<style>
[data-testid="stSidebar"] .stRadio label p {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: white !important;
}

[data-testid="stSidebar"] .stRadio > label {
    font-size: 24px !important;
    font-weight: 800 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

options = st.sidebar.radio("Pages",options=['Overview','Classification','EDA','Model Comparison','Prediction Model'])

#Set formatting for sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: transparent !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: rgba(15, 35, 75, 0.45) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-right: 1px solid rgba(120,180,255,0.12);
    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)

def overview():
  st.markdown("""
  <style>

  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Poppins', sans-serif;
  }

  .stApp {
      background-image:
          linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.42)),
          url("https://images.unsplash.com/photo-1599604770340-c6f8fc3a6585?q=80&w=3131&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      color: white;
  }

  .stApp::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Flag_of_Chicago%2C_Illinois.svg/2560px-Flag_of_Chicago%2C_Illinois.svg.png");
      background-size: cover;
      background-position: center;
      opacity: 0.04;
      z-index: -1;
  }

  header, footer, #MainMenu {
      visibility: visible;
  }

  .block-container {
      padding-top: 2rem;
      max-width: 1250px;
  }

  .hero-box {
      background: rgba(15, 35, 75, 0.72);
      backdrop-filter: blur(14px);
      padding: 60px;
      border-radius: 28px;
      border: 1px solid rgba(120,180,255,0.12);
      box-shadow: 0 0 40px rgba(0,0,0,0.35);
      margin-bottom: 35px;
  }

  .main-title {
      font-size: 4.2rem;
      font-weight: 700;
      color: #ff6666;
      margin-bottom: 15px;
  }

  .subtitle {
      font-size: 1.2rem;
      color: #dbeafe;
      line-height: 1.9;
  }

  .metric-box {
      background: linear-gradient(
          135deg,
          rgba(90, 170, 255, 0.95),
          rgba(30, 90, 180, 0.95)
      );
      padding: 28px;
      border-radius: 22px;
      text-align: center;
      box-shadow: 0 0 25px rgba(70,130,200,0.25);
      margin-bottom: 20px;
      height: 320px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
  }

  .metric-number {
      font-size: 2.1rem;
      font-weight: 700;
      color: white;
      word-break: break-word;
      width: 100%;
  }

  .metric-label {
      font-size: 1rem;
      color: #eef7ff;
  }

  .section-title {
      font-size: 2rem;
      font-weight: 600;
      color: #7ec8ff;
      margin-top: 35px;
      margin-bottom: 20px;
  }

  [data-testid="stDataFrame"] {
      background-color: rgba(15, 35, 75, 0.72);
      border-radius: 18px;
      padding: 10px;
  }

  .footer {
      text-align: center;
      color: #cbd5e1;
      margin-top: 45px;
      padding-bottom: 20px;
  }

  p, li {
      font-size: 1rem;
      line-height: 1.9;
      color: #e5e7eb;
  }

  h1, h2, h3 {
      color: white;
  }

  </style>
  """, unsafe_allow_html=True)

  st.markdown("""
  <div class="hero-box">
  <div class="main-title">👮🏻‍♀️ <u>Chicago Crime ML Dashboard</u></div>
  <div class="subtitle">
  An interactive machine learning and crime analytics dashboard built using the Chicago Crimes dataset from the Chicago Open Data Portal.

  This project explores models that predict the risk level of crime based on location, community type, domestic involvement, arrest status, day, and time period, enabling accurate classification of crime risk patterns.
  </div>
  </div>
  """, unsafe_allow_html=True)

  col1, col2, col3 = st.columns(3)

  with col1:
      st.markdown("""
      <div class="metric-box">
          <div class="metric-number">2017–2026</div>
          <div class="metric-label">Dataset Range</div>
      </div>
      """, unsafe_allow_html=True)

  with col2:
      st.markdown("""
      <div class="metric-box">
          <div class="metric-number">CatBoost | LightGBM | Extra Trees</div>
          <div class="metric-label">Prediction Models</div>
      </div>
      """, unsafe_allow_html=True)

  with col3:
      st.markdown("""
      <div class="metric-box">
          <div class="metric-number">
      <a href="https://data.cityofchicago.org/Public-Safety/Crimes-2026/f6bk-yv3r/about_data" target="_blank" style="color: white; word-break: break-word;">
          https://data.cityofchicago.org/Public-Safety/Crimes-2026/f6bk-yv3r/about_data
      </a>
  </div>
          <div class="metric-label">Police Data Source</div>
      </div>
      """, unsafe_allow_html=True)

  st.markdown("""
  <div class="hero-box">
  <div class="section-title">📌 Project Objective</div>
  <div class="subtitle">
  This project focuses on developing a machine learning classification system to predict crime risk levels using historical crime incident data from the Chicago Police Department.

  The main objectives of this project include:
  - Develop three models to predict crime risk levels accurately and compare their effectiveness.
  - Analyze how features influence risk classification.
  - Identify hidden crime patterns and behavioral trends across different risk categories.
  - Evaluate model performance using classification metrics.
  - Generate data-driven insights that can support predictive crime analytics and public safety decision-making.
  </div>
  </div>
  """, unsafe_allow_html=True)

  st.markdown("""
  <div class="hero-box">
  <div class="section-title">📂 Dataset Information</div>
  <div class="subtitle">
  The dataset originates from the official Chicago Open Data Portal and contains reported crime incidents recorded across multiple years.

  Dataset Includes:
  - Crime categories
  - Arrest records
  - Domestic incident indicators
  - Police districts
  - Location descriptions
  - Date and time information

  Location information is anonymized for privacy protection
  </div>
  </div>
  """, unsafe_allow_html=True)

  st.markdown('<div class="section-title">🧾 Dataset Preview</div>', unsafe_allow_html=True)

  st.dataframe(
      df2.head(10),
      use_container_width=True,
      hide_index=True
  )

  st.markdown('<div class="section-title">📖 Column Legend</div>', unsafe_allow_html=True)

  column_descriptions = {
      "ID": "Unique identifier for the record",
      "Case Number": "Chicago Police Department records division incident number",
      "Date": "Date when the incident occurred (may be estimated)",
      "Block": "Partially redacted address of the incident location",
      "IUCR": "Illinois Uniform Crime Reporting code linked to crime classification",
      "Primary Type": "Primary category of the reported crime",
      "Description": "Detailed subcategory description of the crime",
      "Location Description": "Type of location where the incident occurred",
      "Arrest": "Indicates whether an arrest was made",
      "Domestic": "Indicates whether the incident was domestic-related",
      "Beat": "Smallest police patrol geographic unit",
      "District": "Chicago police district where the incident occurred",
      "Ward": "Chicago city council ward of the incident",
      "Community Area": "One of Chicago's 77 designated community areas",
      "FBI Code": "FBI/NIBRS crime classification code",
      "X Coordinate": "Geographic X coordinate of incident location",
      "Y Coordinate": "Geographic Y coordinate of incident location",
      "Year": "Year the incident was recorded",
      "Updated On": "Last dataset update timestamp",
      "Latitude": "Latitude coordinate of incident location",
      "Longitude": "Longitude coordinate of incident location",
      "Location": "Combined geographic point location"
  }

  legend_df = pd.DataFrame({
      "Column": df2.columns,
      "Meaning": [
          column_descriptions.get(col, "No description available")
          for col in df2.columns
      ]
  })

  st.dataframe(
      legend_df,
      use_container_width=True,
      hide_index=True
  )

  st.markdown("""
  <div class="footer">
  Chicago Crime ML Dashboard • Streamlit Project • City of Chicago Open Data
  </div>
  """, unsafe_allow_html=True)

###############################################################################################################################################################################################

def risk_classification(df_model):

    st.markdown("""
    <style>

    .risk-hero {
        background: rgba(15, 35, 75, 0.72);
        backdrop-filter: blur(16px);
        padding: 50px;
        border-radius: 28px;
        border: 1px solid rgba(120,180,255,0.15);
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
        margin-bottom: 30px;
    }

    .risk-card {
        background: rgba(20, 45, 90, 0.58);
        backdrop-filter: blur(14px);
        padding: 28px;
        border-radius: 22px;
        border: 1px solid rgba(140,200,255,0.12);
        box-shadow: 0 0 20px rgba(0,0,0,0.25);
        text-align: center;
        transition: all 0.3s ease;
    }

    .risk-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 0 28px rgba(120,180,255,0.25);
    }

    .risk-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }

    .risk-sub {
        font-size: 1.1rem;
        color: #dbeafe;
        line-height: 1.8;
    }

    .metric-big {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
    }

    .metric-label {
        color: #dbeafe;
        font-size: 1rem;
    }

    .section-heading {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-top: 35px;
        margin-bottom: 20px;
    }


    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="risk-hero">
        <div class="risk-title"><u>Crime Risk Classification Methodology</u></div>
        <div class="risk-sub">
            Crime incidents were grouped into risk levels based on severity,
            violence potential, and public safety impact to create an interpretable
            machine learning classification target.
        </div>
    </div>
    """, unsafe_allow_html=True)

    low_crimes = [
        "THEFT",
        "DECEPTIVE PRACTICE",
        "CRIMINAL TRESPASS",
        "LIQUOR LAW VIOLATION",
        "PUBLIC PEACE VIOLATION"
    ]

    medium_crimes = [
        "BURGLARY",
        "MOTOR VEHICLE THEFT",
        "NARCOTICS",
        "OTHER OFFENSE",
        "CRIMINAL DAMAGE",
        "PROSTITUTION"
    ]

    high_crimes = [
        "BATTERY",
        "ASSAULT",
        "ROBBERY",
        "WEAPONS VIOLATION",
        "STALKING",
        "INTERFERENCE WITH PUBLIC OFFICER",
        "OFFENSE INVOLVING CHILDREN",
        "INTIMIDATION",
        "HOMICIDE",
        "KIDNAPPING",
        "ARSON",
        "SEX OFFENSE",
        "CRIMINAL SEXUAL ASSAULT",
        "CRIM SEXUAL ASSAULT",
        "CONCEALED CARRY LICENSE VIOLATION"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="risk-card">
            <div class="metric-big">🟢 LOW RISK</div>
            <div class="metric-label">{len(low_crimes)} Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="risk-card">
            <div class="metric-big">🔵 MEDIUM RISK</div>
            <div class="metric-label">{len(medium_crimes)} Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="risk-card">
            <div class="metric-big">🔴 HIGH RISK</div>
            <div class="metric-label">{len(high_crimes)} Categories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>📊 Risk Class Distribution</div>", unsafe_allow_html=True)

    dist = df_model["risk_level"].value_counts().sort_index()

    label_map = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    pie_df = pd.DataFrame({
        "Risk": [label_map[i] for i in dist.index],
        "Count": dist.values
    })

    fig = px.pie(
        pie_df,
        names="Risk",
        values="Count",
        hole=0.45,
        color="Risk",
        color_discrete_map={
        "Low Risk": "#077507",
        "Medium Risk": "#070775",
        "High Risk": "#750707"
    }
    )

    fig.update_traces(
        textinfo="percent+label",
        pull=[0.03, 0.03, 0.03],
        textfont=dict(
        color="white",
        size=16)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=16),
        showlegend=False,
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-heading'>🧭 Crime-to-Risk Hierarchy</div>", unsafe_allow_html=True)

    risk_data = pd.DataFrame({
        "Crime Type": low_crimes + medium_crimes + high_crimes,
        "Risk Level": (
            ["Low Risk"] * len(low_crimes) +
            ["Medium Risk"] * len(medium_crimes) +
            ["High Risk"] * len(high_crimes)
        )
    })

    sunburst = px.sunburst(
        risk_data,
        path=["Risk Level", "Crime Type"],
        color="Risk Level",
        color_discrete_map={
        "Low Risk": "#077507",
        "Medium Risk": "#070775",
        "High Risk": "#750707"
    }

    )

    sunburst.update_traces(
        hovertemplate="<b>%{label}</b><extra></extra>",
        textfont=dict(
        color="white",
        size=20
    ),
       insidetextorientation="radial"
    )

    sunburst.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=18),
        margin=dict(t=10, l=10, r=10, b=10),
        height=1100
    )

    left, center, right = st.columns([1, 6, 1])

    with center:
        st.plotly_chart(sunburst, use_container_width=True)
###############################################################################################################################################################################################

def eda_dashboard(df_model):

    st.markdown("""
    <style>
    .eda-hero {
        background: rgba(15, 35, 75, 0.72);
        backdrop-filter: blur(16px);
        padding: 45px;
        border-radius: 28px;
        border: 1px solid rgba(120,180,255,0.15);
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
        margin-bottom: 30px;
    }

    .eda-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }

    .eda-sub {
        font-size: 1.1rem;
        color: #dbeafe;
        line-height: 1.8;
        margin-top: 10px;
    }

    .chart-box {
        background: rgba(20, 45, 90, 0.55);
        backdrop-filter: blur(14px);
        padding: 20px;
        border-radius: 24px;
        border: 1px solid rgba(140,200,255,0.10);
        box-shadow: 0 0 25px rgba(0,0,0,0.25);
        margin-bottom: 25px;
    }

    .insight-box {
        background: rgba(8, 25, 55, 0.65);
        padding: 14px;
        border-radius: 16px;
        border-left: 4px solid #60a5fa;
        color: #dbeafe;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="eda-hero">
        <div class="eda-title">📊 Exploratory Data Analysis</div>
        <div class="eda-sub">
            This exploratory analysis examines temporal, spatial, and behavioural crime patterns
            to uncover meaningful trends prior to predictive modelling.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = df_model.copy()

    risk_names = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    colors = {
        "Low Risk": "#077507",
        "Medium Risk": "#070775",
        "High Risk": "#750707"
    }

    # ================= ROW 1 =================

    col1, col2 = st.columns(2)

    with col1:

        day_ct = pd.crosstab(
            df["Day"],
            df["risk_level"],
            normalize="index"
        ) * 100

        day_ct = day_ct.rename(columns=risk_names)

        day_order = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        day_ct = day_ct.reindex(day_order)

        fig_day = px.bar(
            day_ct,
            barmode="stack",
            title="Crime Risk Percentage by Day",
            color_discrete_map=colors
        )

        fig_day.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_day, use_container_width=True)

        peak_day = day_ct["High Risk"].idxmax()
        peak_value = day_ct["High Risk"].max()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> <b>{peak_day}</b> records the highest proportion of high-risk crime
        at <b>{peak_value:.1f}%</b>, suggesting severe crime activity becomes more concentrated
        on this day compared to the rest of the week.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        time_ct = pd.crosstab(
            df["time_period"],
            df["risk_level"],
            normalize="index"
        ) * 100

        time_ct = time_ct.rename(columns=risk_names)

        fig_time = px.bar(
            time_ct,
            barmode="stack",
            title="Crime Risk Percentage by Time Period",
            color_discrete_map=colors
        )

        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_time, use_container_width=True)

        peak_period = time_ct["High Risk"].idxmax()
        peak_period_val = time_ct["High Risk"].max()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> The <b>{peak_period}</b> period shows the highest high-risk crime
        concentration at <b>{peak_period_val:.1f}%</b>, indicating elevated severity during this timeframe.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ROW 2 =================

    col3, col4 = st.columns(2)

    with col3:

        hourly = df["Hour"].value_counts().sort_index()

        fig_hour = px.line(
            x=hourly.index,
            y=hourly.values,
            title="Hourly Crime Trend",
            markers=True
        )

        fig_hour.update_traces(line=dict(width=4))

        fig_hour.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_hour, use_container_width=True)

        peak_hour = hourly.idxmax()
        peak_count = hourly.max()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> Crime volume peaks at <b>{peak_hour}:00</b> with approximately
        <b>{peak_count:,}</b> incidents, highlighting a strong temporal clustering effect.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col4:

        community = (
            df["COMMUNITY"]
            .value_counts()
            .head(15)
            .sort_values()
        )

        fig_comm = px.bar(
            x=community.values,
            y=community.index,
            orientation="h",
            title="Top 15 Communities by Crime Count"
        )

        fig_comm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_comm, use_container_width=True)

        hotspot = community.idxmax()
        hotspot_count = community.max()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> <b>{hotspot}</b> emerges as the most significant geographic hotspot
        with <b>{hotspot_count:,}</b> recorded incidents.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ROW 3 =================

    col5, col6 = st.columns(2)

    with col5:

        arrest_rate = (
            df.groupby("risk_level")["Arrest"]
            .mean()
            .mul(100)
            .round(2)
        )

        arrest_df = pd.DataFrame({
            "Risk Level": arrest_rate.index.map(risk_names),
            "Rate": arrest_rate.values
        })

        fig_arrest = px.bar(
            arrest_df,
            x="Risk Level",
            y="Rate",
            title="Arrest Rate by Risk Category",
            color="Risk Level",
            color_discrete_map=colors
        )

        fig_arrest.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500,
            showlegend=False
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_arrest, use_container_width=True)

        arrest_gap = arrest_rate.max() - arrest_rate.min()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> Arrest probability rises substantially with crime severity,
        with a spread of <b>{arrest_gap:.1f}%</b> between the lowest and highest risk groups.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col6:

        domestic_rate = (
            df.groupby("risk_level")["Domestic"]
            .mean()
            .mul(100)
            .round(2)
        )

        domestic_df = pd.DataFrame({
            "Risk Level": domestic_rate.index.map(risk_names),
            "Rate": domestic_rate.values
        })

        fig_domestic = px.bar(
            domestic_df,
            x="Risk Level",
            y="Rate",
            title="Domestic Incident Rate by Risk Category",
            color="Risk Level",
            color_discrete_map=colors
        )

        fig_domestic.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500,
            showlegend=False
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_domestic, use_container_width=True)

        domestic_peak = domestic_rate.idxmax()
        domestic_peak_name = risk_names[domestic_peak]
        domestic_peak_val = domestic_rate.max()

        st.markdown(f"""
        <div class="insight-box">
        🔍 <b>Insight:</b> Domestic incidents are most strongly associated with
        <b>{domestic_peak_name}</b> crimes, reaching <b>{domestic_peak_val:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
###############################################################################################################################################################################################
def model_comparison_dashboard():

    st.markdown("""
    <style>
    .model-hero {
        background: rgba(15, 35, 75, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 50px;
        border-radius: 28px;
        border: 1px solid rgba(120,180,255,0.15);
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
        margin-bottom: 35px;
    }

    .model-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }

    .model-sub {
        font-size: 1.1rem;
        color: #dbeafe;
        line-height: 1.8;
        margin-top: 10px;
    }

    .winner-box {
        background: linear-gradient(
            135deg,
            rgba(8,40,95,0.9),
            rgba(15,80,160,0.65)
        );
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(140,200,255,0.18);
        box-shadow: 0 0 30px rgba(0,120,255,0.18);
        margin-bottom: 35px;
    }

    .winner-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
    }

    .winner-sub {
        color: #dbeafe;
        font-size: 1rem;
        line-height: 1.7;
    }

    .metric-card {
        background: rgba(20, 45, 90, 0.58);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 28px;
        border-radius: 22px;
        border: 1px solid rgba(140,200,255,0.12);
        box-shadow: 0 0 20px rgba(0,0,0,0.25);
        text-align: center;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 0 24px rgba(120,180,255,0.20);
    }

    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    .ranking-card {
        background: rgba(20, 45, 90, 0.55);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(140,200,255,0.10);
        margin-bottom: 18px;
    }

    div[data-testid="stAlert"] {
        border-radius: 18px;
        border: 1px solid rgba(140,200,255,0.12);
        backdrop-filter: blur(10px);
        background: rgba(10, 30, 65, 0.82) !important;
    }

    div[data-testid="stAlert"] * {
        color: white !important;
        font-size: 1.45rem !important;
        line-height: 1.8 !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    data = pd.DataFrame({
        "Model": ["LightGBM", "Extra Trees", "CatBoost"],
        "Accuracy": [0.546277, 0.542320, 0.538367],
        "Macro F1": [0.545725, 0.541858, 0.538774],
        "Weighted F1": [0.546720, 0.543329, 0.539080],
        "Overall Score": [0.546241, 0.542502, 0.538740]
    })

    st.markdown("""
    <div class="model-hero">
        <div class="model-title">🏆 Model Performance Comparison</div>
        <div class="model-sub">
            Three ensemble learning models were evaluated for multiclass crime risk classification.
            Comparative analysis focuses on predictive consistency, balanced multiclass performance,
            and deployment suitability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="winner-box">
        <div class="winner-title">🚀 Best Performing Model: LightGBM</div>
        <div class="winner-sub">
            LightGBM achieved the highest overall performance across all evaluation metrics,
            including accuracy, macro F1, weighted F1, and composite overall score.
            Its balanced multiclass performance makes it the strongest deployment candidate.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    for col, idx in zip([c1, c2, c3], range(3)):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:white;">{data.iloc[idx]['Model']}</h2>
                <p style="color:#dbeafe;">Accuracy: {data.iloc[idx]['Accuracy']:.4f}</p>
                <p style="color:#dbeafe;">Macro F1: {data.iloc[idx]['Macro F1']:.4f}</p>
                <p style="color:#dbeafe;">Weighted F1: {data.iloc[idx]['Weighted F1']:.4f}</p>
                <p style="color:#dbeafe;">Overall Score: {data.iloc[idx]['Overall Score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Performance Metric Comparison</div>", unsafe_allow_html=True)

    long_df = data.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Grouped Metric Comparison",
        color_discrete_map={
            "LightGBM": "#3b82f6",
            "Extra Trees": "#10b981",
            "CatBoost": "#ef4444"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=15),
        height=550,
        yaxis=dict(range=[0.535, 0.548]),
        legend=dict(
            font=dict(size=20, color="white"),
            title=dict(font=dict(size=22, color="white"))
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    GROUPED METRIC ANALYSIS

    LightGBM performs slightly better than the other models across all evaluation metrics, 
    including accuracy, macro F1, weighted F1, and overall score.

    Although the differences are not large, the chart shows that LightGBM consistently stays ahead, 
    while CatBoost records the lowest performance among the three.

    Extra Trees performs closely behind LightGBM, indicating that it is also a strong model. 
    However, since LightGBM achieves the best results across every metric, it was selected as the preferred model for this project.
""")

    st.markdown("<div class='section-title'>🕸 Radar Performance Analysis</div>", unsafe_allow_html=True)

    radar = go.Figure()

    metrics = ["Accuracy", "Macro F1", "Weighted F1", "Overall Score"]

    colors = {
        "LightGBM": "#3b82f6",
        "Extra Trees": "#10b981",
        "CatBoost": "#ef4444"
    }

    for _, row in data.iterrows():
        radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=row["Model"],
            line=dict(color=colors[row["Model"]], width=3),
            opacity=0.55
        ))

    radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0.535, 0.548],
                color="white",
                tickfont=dict(size=13)
            ),
            angularaxis=dict(
                tickfont=dict(size=15, color="white")
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=15),
        height=700,
        legend=dict(
            font=dict(size=20, color="white"),
            title=dict(font=dict(size=22, color="white"))
        )
    )

    st.plotly_chart(radar, use_container_width=True)

    st.info("""
RADAR PERFORMANCE ANALYSIS

LightGBM covers the largest area in the radar chart, 
showing that it performs the most consistently across all evaluation metrics.

Extra Trees shows a very similar pattern and remains competitive, 
but its scores are slightly lower across most metrics.

CatBoost has the smallest overall area, 
which indicates comparatively lower performance among the three models.

Overall, the radar chart supports the earlier comparison results, 
showing that LightGBM provides the strongest and most balanced performance for this classification task.
""")

    st.markdown("<div class='section-title'>🏁 Final Ranking</div>", unsafe_allow_html=True)

    ranking = data.sort_values("Overall Score", ascending=False).reset_index(drop=True)

    medals = ["🥇", "🥈", "🥉"]

    for i, row in ranking.iterrows():
        st.markdown(f"""
        <div class="ranking-card">
            <h2 style="color:white;">{medals[i]} {row['Model']}</h2>
            <p style="color:#dbeafe; font-size:1.1rem;">
                Overall Score: {row['Overall Score']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)

###############################################################################################################################################################################################
def prediction_dashboard():

    @st.cache_resource
    def load_model():
        return joblib.load("lightgbm_model.pkl")

    @st.cache_resource
    def load_encoders():
        return joblib.load("label_encoders.pkl")

    @st.cache_data
    def load_data():
        return pd.read_pickle("df_sample.pkl")

    @st.cache_data
    def load_geojson():
        return joblib.load("chicago_geojson.pkl")

    lightgbm_model = load_model()
    label_encoders = load_encoders()
    df_sample = load_data()
    geojson = load_geojson()

    features = [
        "Location_Group",
        "COMMUNITY",
        "Domestic",
        "Arrest",
        "Day",
        "time_period"
    ]

    risk_labels = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    risk_colors = {
        "Low Risk": "#16a34a",
        "Medium Risk": "#2563eb",
        "High Risk": "#dc2626"
    }

    st.markdown("""
    <style>
    .predict-hero {
        background: rgba(15, 35, 75, 0.78);
        backdrop-filter: blur(16px);
        padding: 45px;
        border-radius: 28px;
        border: 1px solid rgba(120,180,255,0.15);
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
        margin-bottom: 30px;
    }

    .predict-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }

    .predict-sub {
        font-size: 1.1rem;
        color: #dbeafe;
        line-height: 1.8;
        margin-top: 10px;
    }

    .result-card {
        padding: 35px;
        border-radius: 26px;
        text-align: center;
        color: white;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 0 35px rgba(0,0,0,0.35);
    }

    .stat-card {
        background: rgba(20, 45, 90, 0.58);
        backdrop-filter: blur(14px);
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-top: 30px;
        margin-bottom: 20px;
    }

    div[data-testid="stAlert"] {
        background: rgba(10, 30, 65, 0.82) !important;
        border-radius: 18px;
    }

    div[data-testid="stAlert"] * {
        color: white !important;
        font-size: 1.15rem !important;
        line-height: 1.8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="predict-hero">
        <div class="predict-title">Crime Risk Predictor</div>
        <div class="predict-sub">
            Predict crime risk level using the trained LightGBM classification model
            with real-time contextual analysis and community mapping.
        </div>
    </div>
    """, unsafe_allow_html=True)

    location_options = sorted(df_sample["Location_Group"].dropna().astype(str).unique())
    community_options = sorted(df_sample["COMMUNITY"].dropna().astype(str).unique())
    domestic_options = sorted(df_sample["Domestic"].dropna().unique())
    arrest_options = sorted(df_sample["Arrest"].dropna().unique())
    day_options = sorted(df_sample["Day"].dropna().astype(str).unique())
    time_options = sorted(df_sample["time_period"].dropna().astype(str).unique())

    st.markdown("<div class='section-title'>Case Input</div>", unsafe_allow_html=True)

    location = st.selectbox("Location", location_options)

    community = st.selectbox("Community", community_options)

    map_df = pd.DataFrame({
        "COMMUNITY": community_options
    })

    map_df["selected"] = map_df["COMMUNITY"].apply(
        lambda x: 1 if x == community else 0
    )

    map_fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="COMMUNITY",
        featureidkey="properties.community",
        color="selected",
        color_continuous_scale=[
            [0, "#1e293b"],
            [1, "#3b82f6"]
        ],
        mapbox_style="carto-darkmatter",
        zoom=9.7,
        center={"lat": 41.8781, "lon": -87.6298},
        opacity=0.8
    )

    map_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )

    st.plotly_chart(map_fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        domestic = st.selectbox("Domestic", domestic_options)
        day = st.selectbox("Day", day_options)

    with c2:
        arrest = st.selectbox("Arrest", arrest_options)
        time_period = st.selectbox("Time", time_options)

    if st.button("Predict Risk Level", use_container_width=True):

        new_case = pd.DataFrame([{
            "Location_Group": location,
            "COMMUNITY": community,
            "Domestic": domestic,
            "Arrest": arrest,
            "Day": day,
            "time_period": time_period
        }])

        categorical_cols = [
            "Location_Group",
            "COMMUNITY",
            "Day",
            "time_period"
        ]

        for col in categorical_cols:
            new_case[col] = label_encoders[col].transform(
                new_case[col].astype(str)
            )

        new_case["Domestic"] = bool(int(new_case["Domestic"].iloc[0]))
        new_case["Arrest"] = bool(int(new_case["Arrest"].iloc[0]))

        prediction = lightgbm_model.predict(new_case)[0]
        probabilities = lightgbm_model.predict_proba(new_case)[0]
        confidence = np.max(probabilities) * 100

        predicted_label = risk_labels[prediction]
        result_color = risk_colors[predicted_label]

        st.markdown(f"""
        <div class="result-card" style="
            background: linear-gradient(
                135deg,
                {result_color},
                rgba(15,35,75,0.8)
            );
        ">
            <h1>{predicted_label}</h1>
            <h2>Confidence: {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Prediction Confidence Breakdown</div>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
            "Probability": probabilities * 100
        })

        fig = px.bar(
            prob_df,
            x="Risk Level",
            y="Probability",
            color="Risk Level",
            color_discrete_map=risk_colors
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=16),
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Prediction Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result_color}
            }
        ))

        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400
        )

        st.plotly_chart(gauge, use_container_width=True)

        domestic_bool = bool(int(domestic))
        arrest_bool = bool(int(arrest))
        matching_cases = df_sample[
            (df_sample["Location_Group"].astype(str) == location) &
            (df_sample["COMMUNITY"].astype(str) == community) &
            (df_sample["Domestic"] == domestic_bool) &
            (df_sample["Arrest"] == arrest_bool) &
            (df_sample["Day"].astype(str) == day) &
            (df_sample["time_period"].astype(str) == time_period)
        ]

        st.markdown("<div class='section-title'>Historical Context</div>", unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{len(matching_cases)}</h2>
                <p>Similar Historical Cases</p>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{location}</h2>
                <p>Location</p>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{community}</h2>
                <p>Community</p>
            </div>
            """, unsafe_allow_html=True)

        if "risk_level" in matching_cases.columns and len(matching_cases) > 0:

            dist = matching_cases["risk_level"].value_counts().sort_index()

            hist_df = pd.DataFrame({
                "Risk": [risk_labels[i] for i in dist.index],
                "Count": dist.values
            })

            pie = px.pie(
                hist_df,
                names="Risk",
                values="Count",
                hole=0.45,
                color="Risk",
                color_discrete_map=risk_colors
            )

            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=550
            )

            st.plotly_chart(pie, use_container_width=True)

#Page selector

if options == 'Overview':
  overview()
if options == 'Classification':
  risk_classification(df_model)
if options == 'EDA':
  eda_dashboard(df_model)
if options == 'Model Comparison':
  model_comparison_dashboard()
if options == 'Prediction Model':
  prediction_dashboard()

