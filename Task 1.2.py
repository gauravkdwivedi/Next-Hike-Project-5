# Import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_excel('C:/Users/gaura/OneDrive/Visual Studio/NextHike Project 5/users_behavior_data.xlsx')
print(df)

# Remove undwated column
df = df.drop(['Unnamed: 0'], axis=1)

# Check dataset information
df.info()

# Check the description of the data
df.describe()

# Check the outliers in Number of xDR session
sns.boxplot(x=df['number of xDR sessions'])

# Add title and labels
plt.title('Boxplot of Number of xDR Sessions')
plt.xlabel('Number of xDR sessions')
plt.ylabel('Frequency')

# Display the plot
plt.show()

# Check the outliers in Duration of the xDR session
plt.hist(df['Dur. (ms)'], bins='auto')

# Add title and labels
plt.title('Histogram of Duration of xDR Sessions')
plt.xlabel('Duration of xDR sessions')
plt.ylabel('Frequency')

# Display the plot
plt.show()

# Check the outliers in data downloaded and uploaded)
plt.scatter(df['Total DL (Bytes)'], df['Total UL (Bytes)'])

# Add labels and title
plt.xlabel('Total Download (Bytes)')
plt.ylabel('Total Upload (Bytes)')
plt.title('Scatter Plot of Downloaded vs Uploaded Data')

# Display the plot
plt.show()

# Check the Z-scores for each application
selected_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 
                    'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                    'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']

# Calculate Z-Scores for the selected columns
z_scores_selected_columns = df[selected_columns].apply(zscore)

# Plot Z-scores for each selected column
for column in z_scores_selected_columns.columns:
    plt.plot(z_scores_selected_columns[column], 'o', label=column)

plt.xlabel('Data Points')
plt.ylabel('Z-scores')
plt.title('Z-scores for Selected Columns')
plt.legend()
plt.show()

# The dataset contains numerous outliers, and due to a lack of sufficient domain knowledge, it is challenging to discern whether these outliers represent valid data points or errors.
# Careful consideration and consultation with domain experts may be necessary to make informed decisions on how to handle these outliers effectively.

# For Slide
# The 'user_behavior_data' dataset comprises 21 columns, each exclusively containing numeric values.
# The initial column is the 'Bearer ID,' serving as the user identifier.
# Additionally, the dataset includes columns indicating session duration and data downloaded and uploaded through various applications.


## Calculate the basic matrics

# Calculate mean
mean_value = df['number of xDR sessions'].mean()

# Calculate median
median_value = df['number of xDR sessions'].median()

# Calculate mode
mode_value = df['number of xDR sessions'].mode()

# Print the results
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")

# Calculate mean
mean_value = df['Dur. (ms)'].mean()

# Calculate median
median_value = df['Dur. (ms)'].median()

# Calculate mode
mode_value = df['Dur. (ms)'].mode()

# Print the results
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")

# Calculate mean
mean_value = df['Total DL (Bytes)'].mean()

# Calculate median
median_value = df['Total DL (Bytes)'].median()

# Calculate mode
mode_value = df['Total DL (Bytes)'].mode()

# Print the results
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")

# Calculate mean
mean_value = df['Total UL (Bytes)'].mean()

# Calculate median
median_value = df['Total UL (Bytes)'].median()

# Calculate mode
mode_value = df['Total UL (Bytes)'].mode()

# Print the results
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")


## Calculate dispersion parameters for each quantitative variable

# List of quantitative variables
quant_variables = ['Social Media DL (Bytes)','Social Media UL (Bytes)',
                       'Google DL (Bytes)','Google UL (Bytes)','Email DL (Bytes)','Email UL (Bytes)',
                       'Youtube DL (Bytes)','Youtube UL (Bytes)','Netflix DL (Bytes)','Netflix UL (Bytes)'
                       ,'Gaming DL (Bytes)','Gaming UL (Bytes)','Other DL (Bytes)','Other UL (Bytes)']

# Loop through each column
for column in quant_variables:
    # Select the quantitative variable
    quant_variable = df[column]

    # Calculate dispersion parameters
    range_value = quant_variable.max() - quant_variable.min()
    variance_value = quant_variable.var()
    std_deviation_value = quant_variable.std()
    iqr_value = quant_variable.quantile(0.75) - quant_variable.quantile(0.25)

    # Print the results
    print(f"\nDispersion Parameters for {column}:")
    print(f"Range: {range_value}")
    print(f"Variance: {variance_value}")
    print(f"Standard Deviation: {std_deviation_value}")
    print(f"IQR (Interquartile Range): {iqr_value}")


## Graphical univariate analysis

# Set up subplots
fig, axes = plt.subplots(nrows=len(quant_variables), ncols=2, figsize=(12, 2 * len(quant_variables)))

# Loop through each column
for i, column in enumerate(quant_variables):
    # Histogram
    sns.histplot(df[column], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram for {column}')

    # Boxplot
    sns.boxplot(x=df[column], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot for {column}')

# Adjust layout
plt.tight_layout()
plt.show()


## Bivariate Analysis – relationship between each application & the total DL+UL

# List of applications
applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Set up subplots
fig, axes = plt.subplots(nrows=len(applications), ncols=1, figsize=(8, 4 * len(applications)))

# Loop through each application
for i, app in enumerate(applications):
    # Scatter plot
    sns.scatterplot(x=df[f'{app} DL (Bytes)'] + df[f'{app} UL (Bytes)'],
                    y=df['Total DL (Bytes)'] + df['Total UL (Bytes)'],
                    ax=axes[i])
    
    axes[i].set_title(f'Relationship between {app} and Total DL+UL Data')
    axes[i].set_xlabel(f'{app} Data (DL+UL)')
    axes[i].set_ylabel('Total DL+UL Data')

# Adjust layout
plt.tight_layout()
plt.show()


## Top five decile classes based on the total duration for all sessions

total_duration_column = 'Dur. (ms)'
total_dl_column = 'Total DL (Bytes)'
total_ul_column = 'Total UL (Bytes)'

# Calculate total duration per user
total_duration_per_user = df.groupby('MSISDN/Number')[total_duration_column].sum()

# Calculate total DL and total UL per user
total_dl_per_user = df.groupby('MSISDN/Number')[total_dl_column].sum()
total_ul_per_user = df.groupby('MSISDN/Number')[total_ul_column].sum()

# Combine total duration, total DL, and total UL into a new DataFrame
user_data_combined = pd.DataFrame({
    'Total Duration': total_duration_per_user,
    'Total DL (Bytes)': total_dl_per_user,
    'Total UL (Bytes)': total_ul_per_user
})

# Compute total data (DL+UL) per user
user_data_combined['Total Data Bytes'] = user_data_combined['Total DL (Bytes)'] + user_data_combined['Total UL (Bytes)']

# Compute deciles based on total duration
user_data_combined['Decile'] = pd.qcut(user_data_combined['Total Duration'], q=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=False, precision=0)

# Filter for top five decile classes
top_five_deciles = user_data_combined[user_data_combined['Decile'] >= 5]

# Group by decile and calculate total data (DL+UL) per decile class
total_data_per_decile = top_five_deciles.groupby('Decile')['Total Data Bytes'].sum()

# Display the result
print("Total Data (DL+UL) per Decile Class:")
print(total_data_per_decile)


## Correlation Analysis

columns_of_interest = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
                        'Other DL (Bytes)', 'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                        'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)',
                        'Other UL (Bytes)']

# Select relevant columns
selected_columns = df[columns_of_interest]

# Compute correlation matrix
correlation_matrix = selected_columns.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Compute correlation matrix
correlation_matrix = selected_columns.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Selected Columns')
plt.show()


## Dimensionality Reduction - Principal Component Analysis (PCA)

columns_of_interest = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
                        'Other DL (Bytes)', 'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                        'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)',
                        'Other UL (Bytes)']

# Select relevant columns
selected_columns = df[columns_of_interest]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_columns)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Display the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Choose the number of components based on the explained variance
# For example, let's choose the first two components
num_components = 2
selected_pca_result = pca_result[:, :num_components]

# Create a DataFrame for interpretation
pca_df = pd.DataFrame(data=selected_pca_result, columns=[f'PC{i+1}' for i in range(num_components)])

# Display the PCA DataFrame
print("PCA DataFrame:")
print(pca_df)

# The first principal component (PC1) explains approximately 61.30% of the variance, while the second principal component (PC2) explains an additional 3.05%.
# Each subsequent principal component explains a decreasing amount of variance.


## Calculate loadings
loadings = pca.components_
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

loadings_df = pd.DataFrame(data=loadings[:num_components, :].T, columns=[f'PC{i+1}' for i in range(num_components)], index=columns_of_interest)

# Print the Loadings DataFrame
print("Loadings DataFrame:")
print(loadings_df)

# Each row in the DataFrame corresponds to an original feature (e.g., 'Social Media DL (Bytes)', 'Google DL (Bytes)', etc.).
# Each column corresponds to a principal component (e.g., 'PC1' and 'PC2').
# The values in the DataFrame indicate how much each original feature contributes to each principal component. Positive values indicate a positive contribution, and negative values indicate a negative contribution.

# In the first row ('Social Media DL (Bytes)'), you see that it has a positive contribution to 'PC1' (0.267359) and a positive contribution to 'PC2' (0.032938). This means that 'Social Media DL (Bytes)' is positively associated with 'PC1' and 'PC2'.