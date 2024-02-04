# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Load user engagement metrics
user_engagement_metrics = pd.read_excel('user_engagement_metrics.xlsx')
print(user_engagement_metrics)

# Load user experience metrics
user_experience_metrics = pd.read_excel('user_experience_metrics.xlsx')
print(user_experience_metrics)


## Task 4.1 - Assign Engagement and Experience Scores

# Extract cluster centers for engagement clusters
engagement_cluster_centers = user_engagement_metrics.groupby('engagement_cluster')[['Session_Frequency', 'Session_Duration', 'Total_Traffic']].mean()
print(engagement_cluster_centers)

# Assign engagement score as Euclidean distance to the less engaged cluster
user_engagement_metrics['engagement_score'] = distance.cdist(user_engagement_metrics[['Session_Frequency', 'Session_Duration', 'Total_Traffic']], engagement_cluster_centers, 'euclidean').min(axis=1)
print(user_engagement_metrics)

# Extract cluster centers for experience clusters
experience_cluster_centers = user_experience_metrics.groupby('experience_cluster')[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']].mean()
print(experience_cluster_centers)

# Assign experience score as Euclidean distance to the worst experience cluster
user_experience_metrics['experience_score'] = distance.cdist(user_experience_metrics[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']], experience_cluster_centers, 'euclidean').min(axis=1)
print(user_experience_metrics)

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Normalize engagement and experience scores
user_engagement_metrics['engagement_score'] = scaler.fit_transform(user_engagement_metrics[['engagement_score']])
user_experience_metrics['experience_score'] = scaler.fit_transform(user_experience_metrics[['experience_score']])

# Print the normalized scores
print("Normalized Engagement Scores:")
print(user_engagement_metrics[['MSISDN/Number', 'engagement_score']])

print("\nNormalized Experience Scores:")
print(user_experience_metrics[['MSISDN/Number', 'experience_score']])


## Task 4.2 - Calculate Satisfaction Score and Report Top 10 Satisfied Customers

# Calculate satisfaction score as the average of engagement and experience scores
user_satisfaction_metrics = pd.merge(user_engagement_metrics, user_experience_metrics, on='MSISDN/Number', how='inner')
user_satisfaction_metrics['satisfaction_score'] = (user_satisfaction_metrics['engagement_score'] + user_satisfaction_metrics['experience_score']) / 2
print(user_satisfaction_metrics)

# Report top 10 satisfied customers
top_satisfied_customers = user_satisfaction_metrics.nlargest(10, 'satisfaction_score')[['MSISDN/Number', 'satisfaction_score']]
print("Top 10 Satisfied Customers:")
print(top_satisfied_customers)


## Task 4.3 - Build a Regression Model for Satisfaction Score

# Features and target variable
X = user_satisfaction_metrics[['engagement_score', 'experience_score']]
y = user_satisfaction_metrics['satisfaction_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = regression_model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


## Task 4.4 - Run K-Means on Engagement and Experience Scores (k=2)

# Features for k-means clustering
kmeans_features = user_satisfaction_metrics[['engagement_score', 'experience_score']]

# Perform k-means clustering with k=2
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
user_satisfaction_metrics['satisfaction_cluster'] = kmeans_2.fit_predict(kmeans_features)

# Display the updated DataFrame
print(user_satisfaction_metrics[['MSISDN/Number', 'engagement_score', 'experience_score', 'satisfaction_cluster']])


## Task 4.5 - Aggregate Average Satisfaction and Experience Scores per Cluster

# Aggregate average satisfaction and experience scores per cluster
cluster_aggregation = user_satisfaction_metrics.groupby('satisfaction_cluster').agg({
    'satisfaction_score': 'mean',
    'experience_score': 'mean'
})

# Display the result
print("Average Satisfaction and Experience Scores per Cluster:")
print(cluster_aggregation)


# Export the results to a new Excel file
user_satisfaction_metrics.to_excel('user_satisfaction_metrics.xlsx', index=False)