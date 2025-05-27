import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# ===========================
# 1. Load Data and Models
# ===========================

@st.cache_data
def load_crime_data():
    return pd.read_csv('cleaned_data.csv')

@st.cache_resource
def load_crime_model_scaler():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_wellbeing_series():
    df = pd.read_csv('time_series_record.csv')
    df['Time period'] = pd.to_datetime(df['Time period'])
    df.set_index('Time period', inplace=True)
    
    
    # Force numeric columns to numeric dtype
    metrics = [
        'Average Life satisfaction',
        'Average Worthwhile',
        'Average happiness',
        'Average anxiety'
    ]
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')
    return df

@st.cache_data
def load_demographics():
    demo = pd.read_csv('Demographic_recordscsv.csv')
    age_df = demo.iloc[:16].copy()
    gender_df = demo.iloc[16:18].copy()
    region_df = demo.iloc[18:].copy()
    return age_df, gender_df, region_df

# Load once
crime_df = load_crime_data()
kmeans_model, scaler = load_crime_model_scaler()
wb_df = load_wellbeing_series()
age_df, gender_df, region_df = load_demographics()

# ===========================
# 2. Sidebar - Page Selector
# ===========================

page = st.sidebar.selectbox("🔎 Select Dashboard", ["Crime Analysis", "Well-being Time Series"])

# ===========================
# 3. Crime Analysis Page
# ===========================

if page == "Crime Analysis":
    st.title("Crime Data Dashboard with Clustering")

    # KPIs
    total_crimes = len(crime_df)
    total_categories = crime_df['Crime type'].nunique() if 'Crime type' in crime_df.columns else 0
    total_locations = crime_df['Location'].nunique() if 'Location' in crime_df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crimes", total_crimes)
    col2.metric("Total Crime Types", total_categories)
    col3.metric("Total Crime Locations", total_locations)

    # Month Filter
    months = sorted(crime_df['Month'].unique())
    selected_month = st.sidebar.selectbox("Select Month", options=[None] + months)

    if selected_month:
        df_filtered = crime_df[crime_df['Month'] == selected_month]
    else:
        df_filtered = crime_df.copy()

    st.write(f"Showing data for Month: {selected_month}")

    # Monthly Trend of Crime Frequency
    if 'Month' in crime_df.columns:
        st.subheader("Monthly Trend of Crime Frequency")
        monthly_counts = crime_df.groupby('Month').size().reset_index(name='Crime Count')

        fig_trend, ax_trend = plt.subplots(figsize=(16, 7))
        sns.lineplot(data=monthly_counts, x='Month', y='Crime Count', marker='o', ax=ax_trend)
        ax_trend.set_xlabel("Month")
        ax_trend.set_ylabel("Number of Crimes")
        ax_trend.set_title("Trend of Crime Frequency per Month")
        ax_trend.grid(True)
        st.pyplot(fig_trend)
    else:
        st.warning("Column 'Month' not found in data for trend analysis.")

    # Crime Counts by Type
    if 'Crime type' in df_filtered.columns:
        st.subheader("Crime Counts by Type")
        fig1, ax1 = plt.subplots(figsize=(16, 7))
        top_types = df_filtered['Crime type'].value_counts().nlargest(20).index
        sns.countplot(data=df_filtered[df_filtered['Crime type'].isin(top_types)],
                      y='Crime type',
                      order=top_types,
                      ax=ax1)
        st.pyplot(fig1)
    else:
        st.warning("Column 'Crime type' not found in data.")

    # Counts by Last Outcome Category
    try:
        if 'Last outcome category' in df_filtered.columns:
            st.subheader("Counts by Last Outcome Category")
            top_outcomes = df_filtered['Last outcome category'].value_counts().nlargest(20).index
            df_top_outcomes = df_filtered[df_filtered['Last outcome category'].isin(top_outcomes)]

            fig, ax = plt.subplots(figsize=(16, 7))
            sns.countplot(data=df_top_outcomes, x='Last outcome category', order=top_outcomes, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Column 'Last outcome category' not found in data.")
    except Exception as e:
        st.error(f"Error plotting Last outcome category: {e}")

    # Crime Counts by Location (Top 20)
    if 'Location' in df_filtered.columns:
        st.subheader("Crime Counts by Location (Top 20)")
        top_locations = df_filtered['Location'].value_counts().nlargest(20).index
        df_top_loc = df_filtered[df_filtered['Location'].isin(top_locations)]

        fig2, ax2 = plt.subplots(figsize=(16, 7))
        sns.countplot(data=df_top_loc,
                      y='Location',
                      order=top_locations,
                      ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Column 'Location' not found in data.")

    # Crime Counts by Cluster
    if 'Cluster' in df_filtered.columns:
        st.subheader("Crime Counts by Cluster")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        df_filtered['Cluster'].value_counts().plot.pie(autopct='%1.1f%%',
                                                      startangle=90,
                                                      ax=ax3)
        ax3.set_ylabel('')
        st.pyplot(fig3)
    else:
        st.warning("Column 'Cluster' not found in data.")

    # Cluster Prediction Form in Sidebar
    st.sidebar.header("Predict Crime Cluster")

    feature_cols = ['Month', 'Longitude', 'Latitude', 'Crime type', 'Last outcome category', 'Location_Encoded']

    categories_dict = {
        'Crime type': sorted(crime_df['Crime type'].unique()) if 'Crime type' in crime_df.columns else [],
        'Last outcome category': sorted(crime_df['Last outcome category'].unique()) if 'Last outcome category' in crime_df.columns else [],
        'Location': sorted(crime_df['Location'].unique()) if 'Location' in crime_df.columns else [],
    }

    input_data = {}

    if categories_dict['Location']:
        selected_location = st.sidebar.selectbox("Location", categories_dict['Location'])
        input_data['Location_Encoded'] = categories_dict['Location'].index(selected_location)
    else:
        st.sidebar.warning("No location categories available.")
        input_data['Location_Encoded'] = 0

    for col in feature_cols:
        if col == 'Location_Encoded':
            continue
        if col in categories_dict:
            options = categories_dict[col]
            if options:
                val = st.sidebar.selectbox(f"{col}", options)
                val_encoded = options.index(val)
                input_data[col] = val_encoded
            else:
                val = st.sidebar.number_input(f"{col} (numeric)", value=0)
                input_data[col] = val
        else:
            val = st.sidebar.number_input(f"{col}", value=0.0)
            input_data[col] = val

    if st.sidebar.button("Predict Cluster"):
        try:
            input_array = np.array([input_data[col] for col in feature_cols]).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            cluster_pred = kmeans_model.predict(input_scaled)
            st.sidebar.success(f"Predicted Cluster: {cluster_pred[0]}")
        except Exception as e:
            st.sidebar.error(f"Error in prediction: {e}")

# ===========================
# 4. Well-being Time Series Page
# ===========================

else:
    st.title("Well-being Indicator Time Series Analysis")

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Life Satisfaction", f"{wb_df['Average Life satisfaction'].mean():.2f}")
    k2.metric("Avg Worthwhile", f"{wb_df['Average Worthwhile'].mean():.2f}")
    k3.metric("Avg Happiness", f"{wb_df['Average happiness'].mean():.2f}")
    k4.metric("Avg Anxiety", f"{wb_df['Average anxiety'].mean():.2f}")

    # Demographics Bar Grids
    def demo_grid(df, title):
        st.subheader(title)
        metrics = [
            'Average Life satisfaction',
            'Average Worthwhile',
            'Average happiness',
            'Average anxiety'
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for ax, m in zip(axes.flatten(), metrics):
            ax.bar(df['Demographic characteristic'], df[m], alpha=0.8)
            ax.set_title(m)
            ax.set_xticklabels(df['Demographic characteristic'], rotation=45, ha='right')
            ax.set_ylabel("")
            ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    demo_grid(age_df, "By Age Group")
    demo_grid(gender_df, "By Gender")
    demo_grid(region_df, "By Region")

    # Historical Quarterly Time Series Plots
    st.subheader("Well-being Quarterly Time Series (Historical)")
    vars_ = wb_df.columns.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, col in zip(axes.flatten(), vars_):
        ts = wb_df[col].dropna()
        if ts.empty:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        else:
            ax.plot(ts.index, ts.values, marker='o', linestyle='-')
        ax.set_title(col)
        ax.set_xlabel("Quarter")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # SARIMAX Forecasts with differencing orders
    diff_orders = {
        'Average Life satisfaction': 1,
        'Average Worthwhile': 2,
        'Average happiness': 2,
        'Average anxiety': 1,
    }

    st.subheader("10-Quarter Ahead Forecasts (SARIMAX with 95% CI)")
    horizon = 10
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for ax, col in zip(axes.flatten(), vars_):
        ts = wb_df[col].dropna()

        if len(ts) < 8:
            ax.text(0.5, 0.5, "Insufficient data for SARIMAX", ha='center', va='center')
            ax.set_title(col)
            ax.set_xlabel("Quarter")
            ax.set_ylabel(col)
            continue

        d = diff_orders.get(col, 1)

        model = SARIMAX(
            ts,
            order=(1, d, 1),
            seasonal_order=(1, 1, 1, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        res = model.get_forecast(steps=horizon)
        fc = res.predicted_mean
        ci = res.conf_int(alpha=0.05)
        future_idx = pd.period_range(ts.index[-1], periods=horizon + 1, freq='Q')[1:].to_timestamp()

        ax.plot(ts.index, ts.values, marker='o', label='Historical')
        ax.plot(future_idx, fc.values, linestyle='--', label='Forecast')
        ax.fill_between(future_idx, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, label='95% CI')
        ax.set_title(col)
        ax.set_xlabel('Quarter')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
