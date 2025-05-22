import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Load cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/mac/Desktop/jide/cleaned_data.csv')
    return df

# Load KMeans model and scaler
@st.cache_resource
def load_model_scaler():
    with open('/Users/mac/Desktop/jide/kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('/Users/mac/Desktop/jide/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

df = load_data()
model, scaler = load_model_scaler()

# -------------------
# Page title
st.title("Crime Data Dashboard with Clustering")

# -------------------
# KPIs
total_crimes = len(df)
total_categories = df['Crime type'].nunique() if 'Crime type' in df.columns else 0
total_locations = df['Location'].nunique() if 'Location' in df.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Crimes", total_crimes)
col2.metric("Total Crime Types", total_categories)
col3.metric("Total Crime Locations", total_locations)

# -------------------
# Date filter: Month
months = sorted(df['Month'].unique())
selected_month = st.sidebar.selectbox("Select Month", options=[None] + months)

# Filter dataframe based on selected month
if selected_month:
    df_filtered = df[df['Month'] == selected_month]
else:
    df_filtered = df.copy()

st.write(f"Showing data for Month: {selected_month}")

# -------------------
# Visualizations

# Aggregate crime counts per month
if 'Month' in df.columns:
    st.subheader("Monthly Trend of Crime Frequency")
    monthly_counts = df.groupby('Month').size().reset_index(name='Crime Count')
    
    # Plot line chart
    fig_trend, ax_trend = plt.subplots(figsize=(16, 7))
    sns.lineplot(data=monthly_counts, x='Month', y='Crime Count', marker='o', ax=ax_trend)
    ax_trend.set_xlabel("Month")
    ax_trend.set_ylabel("Number of Crimes")
    ax_trend.set_title("Trend of Crime Frequency per Month")
    ax_trend.grid(True)
    st.pyplot(fig_trend)
else:
    st.warning("Column 'Month' not found in data for trend analysis.")


# Crime Counts by type
if 'Crime type' in df_filtered.columns:
    st.subheader("Crime Counts by Type")
    fig1, ax1 = plt.subplots(figsize=(16,7))
    # Show all or top 20 if too many categories
    top_types = df_filtered['Crime type'].value_counts().nlargest(20).index
    sns.countplot(data=df_filtered[df_filtered['Crime type'].isin(top_types)],
                  y='Crime type',
                  order=top_types,
                  ax=ax1)
    st.pyplot(fig1)
else:
    st.warning("Column 'Crime type' not found in data.")

st.subheader("Counts by Last Outcome Category")

try:
    # Limit to top 20 categories for performance
    if 'Last outcome category' in df_filtered.columns:
        top_outcomes = df_filtered['Last outcome category'].value_counts().nlargest(20).index
        df_top_outcomes = df_filtered[df_filtered['Last outcome category'].isin(top_outcomes)]

        fig, ax = plt.subplots(figsize=(16,7))
        sns.countplot(data=df_top_outcomes, x='Last outcome category', order=top_outcomes, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Column 'Last outcome category' not found in data.")
except Exception as e:
    st.error(f"Error plotting Last outcome category: {e}")


# Crime Counts by Location (Top 20 locations only)
if 'Location' in df_filtered.columns:
    st.subheader("Crime Counts by Location (Top 20)")
    top_locations = df_filtered['Location'].value_counts().nlargest(20).index
    df_top_loc = df_filtered[df_filtered['Location'].isin(top_locations)]

    fig2, ax2 = plt.subplots(figsize=(16,7))
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
    fig3, ax3 = plt.subplots(figsize=(6,6))
    df_filtered['Cluster'].value_counts().plot.pie(autopct='%1.1f%%',
                                                  startangle=90,
                                                  ax=ax3)
    ax3.set_ylabel('')
    st.pyplot(fig3)
else:
    st.warning("Column 'Cluster' not found in data.")

# -------------------
# Cluster Prediction Form in Sidebar

st.sidebar.header("Predict Crime Cluster")

feature_cols = ['Month', 'Longitude', 'Latitude', 'Crime type', 'Last outcome category', 'Location_Encoded']

# Original categorical columns for user input
categories_dict = {
    'Crime type': sorted(df['Crime type'].unique()) if 'Crime type' in df.columns else [],
    'Last outcome category': sorted(df['Last outcome category'].unique()) if 'Last outcome category' in df.columns else [],
    'Location': sorted(df['Location'].unique()) if 'Location' in df.columns else [],
}

input_data = {}

# Location input as original category
if categories_dict['Location']:
    selected_location = st.sidebar.selectbox("Location", categories_dict['Location'])
    # Encode to numeric for model input
    input_data['Location_Encoded'] = categories_dict['Location'].index(selected_location)
else:
    st.sidebar.warning("No location categories available.")
    input_data['Location_Encoded'] = 0

# Other inputs
for col in feature_cols:
    if col == 'Location_Encoded':
        continue  # already handled above
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
        # Numeric inputs
        val = st.sidebar.number_input(f"{col}", value=0.0)
        input_data[col] = val

if st.sidebar.button("Predict Cluster"):
    try:
        input_array = np.array([input_data[col] for col in feature_cols]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        cluster_pred = model.predict(input_scaled)
        st.sidebar.success(f"Predicted Cluster: {cluster_pred[0]}")
    except Exception as e:
        st.sidebar.error(f"Error in prediction: {e}")
