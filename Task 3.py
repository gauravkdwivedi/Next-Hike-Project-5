# Import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_excel('C:/Users/gaura/OneDrive/Visual Studio/NextHike Project 5/telcom_data_cleaned.xlsx')

print(df)

# Identify numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Replace NaN values with the mean for numeric columns only
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df.info()

df.isnull().sum()

# There is no use of Last Location Name column. So, we can ignore this column for now.

## Task 3.1

# Average TCP retransmission
avg_tcp_retransmission = df.groupby('MSISDN/Number')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean().mean(axis=1)
avg_tcp_retransmission

# Average RTT
avg_rtt = df.groupby('MSISDN/Number')[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean().mean(axis=1)
avg_rtt

# Handset type
Handset_type = df.groupby('MSISDN/Number')['Handset Type'].unique()
Handset_type

# Average throughput
avg_throughput = df.groupby('MSISDN/Number')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().mean(axis=1)
avg_throughput

# Create a new DataFrame with aggregated information

# Create a DataFrame for 'MSISDN/Number'
MSISDN_Number = df['MSISDN/Number'].unique()

experience_metrics = pd.DataFrame({
    'MSISDN/Number': MSISDN_Number,
    'Avg_TCP_Retransmission': avg_tcp_retransmission,
    'Avg_RTT': avg_rtt,
    'Handset_Type': Handset_type,
    'Avg_Throughput': avg_throughput
})
experience_metrics

experience_metrics.isnull().sum()


## Task 3.2

# Function to compute and list top, bottom, and most frequent values
def compute_and_list_values(values, column_name):
    print(f"Top 10 {column_name} values:")
    print(values.nlargest(10))
    
    print(f"\nBottom 10 {column_name} values:")
    print(values.nsmallest(10))
    
    print(f"\nTop 10 most frequent {column_name} values:")
    print(values.value_counts().nlargest(10))

# Compute and list values for Avg_TCP_Retransmission
print("TCP values:")
compute_and_list_values(experience_metrics['Avg_TCP_Retransmission'], 'Avg_TCP_Retransmission')
print("\n------------------------\n")

# Compute and list values for Avg_RTT
print("RTT values:")
compute_and_list_values(experience_metrics['Avg_RTT'], 'Avg_RTT')
print("\n------------------------\n")

# Compute and list values for Avg_Throughput
print("Throughput values:")
compute_and_list_values(experience_metrics['Avg_Throughput'], 'Avg_Throughput')


## Task 3.3

# 'Handset_Type' should be the first value in the array for each group
Handset_type = df.groupby('MSISDN/Number')['Handset Type'].first()

# Create a new DataFrame with aggregated information
experience_metrics = pd.DataFrame({
    'MSISDN/Number': df['MSISDN/Number'].unique(),
    'Avg_TCP_Retransmission': avg_tcp_retransmission,
    'Avg_RTT': avg_rtt,
    'Handset_Type': Handset_type,
    'Avg_Throughput': avg_throughput
})


# Compute and report the distribution of average throughput per handset type
avg_throughput_by_handset = experience_metrics.groupby('Handset_Type')['Avg_Throughput'].mean()
print("Distribution of Average Throughput per Handset Type:")
print(avg_throughput_by_handset)
print("\n------------------------\n")

# Compute and report the distribution of average TCP retransmission per handset type
avg_tcp_retransmission_by_handset = experience_metrics.groupby('Handset_Type')['Avg_TCP_Retransmission'].mean()
print("Distribution of Average TCP Retransmission per Handset Type:")
print(avg_tcp_retransmission_by_handset)
print("\n------------------------\n")

# Compute and report the top 10 handsets based on average throughput per handset type
top_avg_throughput_by_handset = experience_metrics.groupby('Handset_Type')['Avg_Throughput'].mean().nlargest(10)

# Create a bar chart for top average throughput
plt.figure(figsize=(10, 6))
top_avg_throughput_by_handset.plot(kind='bar', color='skyblue')
plt.title('Top 10 Handsets based on Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.xticks(rotation=45, ha='right')
plt.show()

# Compute and report the top 10 handsets based on average TCP retransmission per handset type
top_avg_tcp_retransmission_by_handset = experience_metrics.groupby('Handset_Type')['Avg_TCP_Retransmission'].mean().nlargest(10)

# Create a bar chart for top average TCP retransmission
plt.figure(figsize=(10, 6))
top_avg_tcp_retransmission_by_handset.plot(kind='bar', color='salmon')
plt.title('Top 10 Handsets based on Average TCP Retransmission per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.xticks(rotation=45, ha='right')
plt.show()


## Task 3.4

# Select relevant columns for clustering
columns_for_clustering = ['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']

# Extract the subset of columns
data_for_clustering = experience_metrics[columns_for_clustering]

# Standardize the features
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
experience_metrics['experience_cluster'] = kmeans.fit_predict(data_for_clustering_scaled)
print(experience_metrics)

# Analyze the clusters
cluster_analysis = experience_metrics.groupby('experience_cluster')[columns_for_clustering].mean()
print("Cluster Analysis:")
print(cluster_analysis)

# Description of each cluster based on understanding of the data
cluster_descriptions = {
    0: "Low retransmission, low RTT, high throughput",
    1: "Medium retransmission, medium RTT, medium throughput",
    2: "High retransmission, high RTT, low throughput"
}

for cluster, description in cluster_descriptions.items():
    print(f"\nCluster {cluster} - {description}")
    print(experience_metrics[experience_metrics['experience_cluster'] == cluster]['MSISDN/Number'])


# Scatter plot for two features
plt.scatter(data_for_clustering_scaled[:, 0], data_for_clustering_scaled[:, 1], c=experience_metrics['experience_cluster'], cmap='viridis')
plt.title('K-Means Clustering - Scatter Plot')
plt.xlabel('Avg_TCP_Retransmission (Standardized)')
plt.ylabel('Avg_RTT (Standardized)')
plt.show()

# Export experience_metrics
experience_metrics[['MSISDN/Number', 'Avg_TCP_Retransmission', 'Avg_RTT', 'Handset_Type', 'Avg_Throughput', 'experience_cluster']].to_excel('C:/Users/gaura/OneDrive/Visual Studio/NextHike Project 5/user_experience_metrics.xlsx', index=False)