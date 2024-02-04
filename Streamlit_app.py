import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Load the cleaned dataset
telcom_data = pd.read_excel('telcom_data_cleaned.xlsx')

# Load the user_engagement_data
user_engagement_data = pd.read_excel('user_engagement_data.xlsx')

# Load user engagement metrics
engagement_metrics = pd.read_excel('user_engagement_metrics.xlsx')

# Load user experience metrics
experience_metrics = pd.read_excel('user_experience_metrics.xlsx')

# Load user satisfaction metrics
satisfaction_metrics = pd.read_excel('user_satisfaction_metrics.xlsx')

# Page setting
st.set_page_config(layout='wide')

# Load CSS style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Title of the web app
# Display the image above the title
st.image('Telco logo.png', width=150)

# Display the title
st.title("Telco")

# Row A
a1, a2, a3, a4 = st.columns(4)

## Display the Top 10 'Handset Type'
top_handsets = telcom_data['Handset Type'].value_counts().head(10)

# Create a vertical bar chart using Plotly
fig_handsets = go.Figure()

fig_handsets.add_trace(go.Bar(
    y=top_handsets.index,
    x=top_handsets.values,
    marker=dict(color='blue')
))

# Update layout for the chart
fig_handsets.update_layout(
    title="Top 10 Handsets used by the customer",
    xaxis_title="Handset",
    yaxis_title="Count",
    height=350,
    width=500
)

# Display the chart for Top 3 handsets
a1.plotly_chart(fig_handsets)


## Display the Top 3 'Handset Manufacturers'
top_manufacturers = telcom_data['Handset Manufacturer'].value_counts().head(3)

# Create a donut chart using Plotly
fig_top_manufacturers = go.Figure()

fig_top_manufacturers.add_trace(go.Pie(
    labels=top_manufacturers.index,
    values=top_manufacturers.values,
    hole=0.4,  # Increase the size of the hole for better centering
    marker=dict(colors=['darkorange', 'gray', 'amber'])
))

# Update layout for the chart
fig_top_manufacturers.update_layout(
    title="Top 3 Handsets Manufacturer",
    height=350,
    width=500
)

# Display the chart for Top 3 handset manufacturers
a3.plotly_chart(fig_top_manufacturers)


# Row B
b1, b2, b3, b4 = st.columns(4)

## Top 5 handsets for Top 3 manufacturers
top_manufacturers = telcom_data['Handset Manufacturer'].value_counts().head(3).index

# Create a horizontal bar chart for each manufacturer using Plotly
fig_manufacturers = go.Figure()

for manufacturer in top_manufacturers:
    # Filter DataFrame for rows with the current manufacturer
    manufacturer_telcom_data = telcom_data[telcom_data['Handset Manufacturer'] == manufacturer]

    # Add a trace for each manufacturer
    fig_manufacturers.add_trace(go.Bar(
        y=manufacturer_telcom_data['Handset Type'].value_counts().head(5).index,
        x=manufacturer_telcom_data['Handset Type'].value_counts().head(5).values,
        orientation='h',
        name=f"Top 5 handsets for {manufacturer}"
    ))

# Update layout for the combined chart
fig_manufacturers.update_layout(
    title="Top 5 handsets for Top 3 manufacturers",
    barmode='group',
    height=350,
    width=500
)

# Display the combined chart for Top 3 manufacturers
b1.plotly_chart(fig_manufacturers)


## Top 3 most used applications charts

# Create a new column 'Total_Traffic' by summing up DL and UL traffic for each application
telcom_data['Total_Traffic'] = telcom_data['Total DL (Bytes)'] + telcom_data['Total UL (Bytes)']

# Create a list of applications
applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Iterate through each application
for app in applications:
    # Create a new column for the specific application's total traffic
    telcom_data[f'{app}_Traffic'] = telcom_data[f'{app} DL (Bytes)'] + telcom_data[f'{app} UL (Bytes)']

# Total traffic per application
total_traffic_per_app = telcom_data[['Social Media_Traffic', 'Google_Traffic', 'Email_Traffic', 'Youtube_Traffic', 'Netflix_Traffic', 'Gaming_Traffic', 'Other_Traffic']]

# Calculate the total traffic for each application
total_traffic_per_app = total_traffic_per_app.sum()

# Get the top 3 applications
top_3_apps = total_traffic_per_app.nlargest(3)

# Plot the top 3 most used applications using Plotly
fig_apps = go.Figure()

fig_apps.add_trace(go.Bar(
    x=top_3_apps.index,
    y=top_3_apps.values,
    marker=dict(color=['blue', 'orange', 'green']),
    orientation='v'
))

# Update layout for the chart
fig_apps.update_layout(
    title="Top 3 most used applications",
    xaxis_title="Applications",
    yaxis_title="Total Traffic (Bytes)",
    height=350,  # Set the height of the chart
    width=500    # Set the width of the chart
)

# Display the chart for Top 3 most used applications
b3.plotly_chart(fig_apps)


# Row C
c1, c2, c3, c4 = st.columns(4)

# Row D
d1, d2, d3, d4 = st.columns(4)

# Row E
e1, e2, e3, e4 = st.columns(4)

## Top 10 customers per engagement metric
top_10_sessions_frequency = user_engagement_data.nlargest(10, 'Session_Frequency')
top_10_session_duration = user_engagement_data.nlargest(10, 'Session_Duration')
top_10_total_traffic = user_engagement_data.nlargest(10, 'Total_Traffic')

# Display top 10 customers per engagement metric
c1.subheader("Top 10 Customers by Session Frequency")
c1.table(top_10_sessions_frequency.reset_index(drop=True))

# Create a histogram for top 10 customers per session frequency
fig_session_frequency_hist = go.Figure()

fig_session_frequency_hist.add_trace(go.Histogram(
    x=top_10_sessions_frequency['MSISDN/Number'],
    y=top_10_sessions_frequency['Session_Frequency'],
    marker=dict(color='purple'),
    nbinsx=10  # Adjust the number of bins as needed
))

# Update layout for the chart
fig_session_frequency_hist.update_layout(
    title="Distribution of Session Frequency for Top 10 Customers",
    xaxis_title="MSISDN/Number",
    yaxis_title="Session Frequency",
    height=400,
    width=500
)

# Display the chart for distribution of session frequency
c3.plotly_chart(fig_session_frequency_hist)

# Display top 10 customers by session duration
d1.header("Top 10 Customers by Session Duration")
d1.table(top_10_session_duration.reset_index(drop=True))

# Create a histogram for top 10 customers by session duration
fig_session_duration_hist = go.Figure()

fig_session_duration_hist.add_trace(go.Histogram(
    x=top_10_session_duration['MSISDN/Number'],
    y=top_10_session_duration['Session_Duration'],
    marker=dict(color='orange'),
    nbinsx=10  # Adjust the number of bins as needed
))

# Update layout for the chart
fig_session_duration_hist.update_layout(
    title="Distribution of Session Duration for Top 10 Customers",
    xaxis_title="MSISDN/Number",
    yaxis_title="Session Duration",
    height=400,
    width=500
)

# Display the chart for distribution of session duration
d3.plotly_chart(fig_session_duration_hist)

# Display top 10 customers by total traffic
e1.header("Top 10 Customers by Total Traffic")
e1.table(top_10_total_traffic.reset_index(drop=True))

# Create a histogram for top 10 customers by total traffic
fig_total_traffic_hist = go.Figure()

fig_total_traffic_hist.add_trace(go.Histogram(
    x=top_10_total_traffic['MSISDN/Number'],
    y=top_10_total_traffic['Total_Traffic'],
    marker=dict(color='green'),
    nbinsx=10  # Adjust the number of bins as needed
))

# Update layout for the chart
fig_total_traffic_hist.update_layout(
    title="Distribution of Total Traffic for Top 10 Customers",
    xaxis_title="MSISDN/Number",
    yaxis_title="Total Traffic",
    height=400,
    width=500
)

# Display the chart for distribution of total traffic
e3.plotly_chart(fig_total_traffic_hist)


# Row F
f1, f2 = st.columns(2)

# Row G
g1, g2 = st.columns(2)

# Row H
h1, h2 = st.columns(2)

## Top 10 customers per engagement metric
top_10_Avg_TCP_Retransmission = experience_metrics.nlargest(10, 'Avg_TCP_Retransmission')[['MSISDN/Number', 'Avg_TCP_Retransmission']]
top_10_Avg_RTT = experience_metrics.nlargest(10, 'Avg_RTT')[['MSISDN/Number', 'Avg_RTT']]
top_10_Avg_Throughput = experience_metrics.nlargest(10, 'Avg_Throughput')[['MSISDN/Number', 'Avg_Throughput']]

# Display top 10 customers per engagement metric
f1.subheader("Top 10 TCP values in the dataset")
f1.table(top_10_Avg_TCP_Retransmission.reset_index(drop=True))

g1.subheader("Top 10 RTT values in the dataset")
g1.table(top_10_Avg_RTT.reset_index(drop=True))

h1.subheader("Top 10 Throughput values in the dataset")
h1.table(top_10_Avg_Throughput.reset_index(drop=True))


# Row I
i1, i2 = st.columns(2)

## Top 10 satisfied customers
top_satisfied_customers = satisfaction_metrics.nlargest(10, 'satisfaction_score')[['MSISDN/Number', 'satisfaction_score']]
i1.subheader("Top 10 Satisfied Customers:")
i1.table(top_satisfied_customers.reset_index(drop=True))