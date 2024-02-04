# Import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset
df = pd.read_excel('user_engagement_data.xlsx')

print(df)

# Top 10 customers per engagement metric
top_10_sessions_frequency = df.nlargest(10, 'Session_Frequency')
top_10_session_duration = df.nlargest(10, 'Session_Duration')
top_10_total_traffic = df.nlargest(10, 'Total_Traffic')

print(top_10_sessions_frequency)
print(top_10_session_duration)
print(top_10_total_traffic)

# Normalize user_engagement_data
scaler = MinMaxScaler()
engagement_metrics_normalized = scaler.fit_transform(df[['Session_Frequency', 'Session_Duration', 'Total_Traffic']])

print(engagement_metrics_normalized)

# Run a k-means (k=3) to classify customers into three groups of engagement
kmeans = KMeans(n_clusters=3, random_state=0)
engagement_clusters = kmeans.fit_predict(engagement_metrics_normalized)

print(engagement_clusters)


# Add engagement clusters to the DataFrame
df['engagement_cluster'] = engagement_clusters

print(df)

# Group by the engagement cluster and calculate metrics
cluster_metrics = df.groupby('engagement_cluster').agg({
    'Session_Frequency': ['min', 'max', 'mean', 'sum'],
    'Session_Duration': ['min', 'max', 'mean', 'sum'],
    'Total_Traffic': ['min', 'max', 'mean', 'sum']
})

print(cluster_metrics)


# Run a k-means (k=3) to classify customers into three groups of engagement without normalizing the data
engagement_metrics_non_normalized = df[['Session_Frequency', 'Session_Duration', 'Total_Traffic']]
kmeans = KMeans(n_clusters=3, random_state=0)
engagement_clusters_non_normalized = kmeans.fit_predict(engagement_metrics_non_normalized)

# Add engagement clusters to the DataFrame
df['engagement_cluster_non_normalized'] = engagement_clusters_non_normalized

# Group by the engagement cluster and calculate metrics
cluster_metrics_non_normalized = df.groupby('engagement_cluster_non_normalized').agg({
    'Session_Frequency': ['min', 'max', 'mean', 'sum'],
    'Session_Duration': ['min', 'max', 'mean', 'sum'],
    'Total_Traffic': ['min', 'max', 'mean', 'sum']
})

print(cluster_metrics_non_normalized)

# Visualize non-normalized metrics for each cluster
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for i, metric in enumerate(['Session_Frequency', 'Session_Duration', 'Total_Traffic']):
    cluster_metrics_non_normalized.plot(kind='bar', ax=axes[i], rot=0)
    axes[i].set_title(f'{metric} Across Clusters')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()


# Aggregate user total traffic per application and derive the top 10 most engaged users per application

# Create a new column 'Total_Traffic' by summing up DL and UL traffic for each application
df['Total_Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

# Create a list of applications
applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Create a dictionary to store the top 10 users per application
top_users_per_app = {}

# Iterate through each application
for app in applications:
    # Create a new column for the specific application's total traffic
    df[f'{app}_Traffic'] = df[f'{app} DL (Bytes)'] + df[f'{app} UL (Bytes)']
    
    # Group by user and sum the total traffic for the current application
    app_traffic_per_user = df.groupby('MSISDN/Number')[f'{app}_Traffic'].sum().reset_index()
    
    # Find the top 10 users for the current application
    top_users = app_traffic_per_user.nlargest(10, f'{app}_Traffic')
    
    # Add the top users to the dictionary
    top_users_per_app[app] = top_users

# Print the top 10 users for each application
for app, top_users in top_users_per_app.items():
    print(f"Top 10 users for {app}:\n{top_users}\n")


# Top 3 most used applications charts.
    
# Total traffic per application
total_traffic_per_app = df[['Social Media_Traffic', 'Google_Traffic', 'Email_Traffic', 'Youtube_Traffic', 'Netflix_Traffic', 'Gaming_Traffic', 'Other_Traffic']]

# Calculate the total traffic for each application
total_traffic_per_app = total_traffic_per_app.sum()

# Get the top 3 applications
top_3_apps = total_traffic_per_app.nlargest(3)

# Plot the top 3 most used applications
plt.figure(figsize=(10, 6))
top_3_apps.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Top 3 Most Used Applications')
plt.xlabel('Applications')
plt.ylabel('Total Traffic (Bytes)')
plt.show()


# Using the k-means clustering algorithm, group users in k engagement clusters

# df contains the engagement metrics
engagement_metrics = df[['Session_Frequency', 'Session_Duration', 'Total_Traffic']]

# Create an empty list to store the sum of squared distances (inertia) for each k
inertia = []

# Define the range of k values
k_values = range(1, 11)

# Run k-means clustering for each k and calculate inertia
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(engagement_metrics)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.show()

# The elbow point in the chart is at k=2, it suggests that having 2 clusters is an optimal choice for k-means clustering algorithm based on the engagement metrics.
# The plot shows a clear elbow point, it suggests that there is a natural grouping in the data, and the optimal k is the value at the elbow.

# Export user_engagement_matrics
df[['MSISDN/Number', 'Session_Frequency', 'Session_Duration', 'Total_Traffic', 'engagement_cluster']].to_excel('user_engagement_metrics.xlsx', index=False)