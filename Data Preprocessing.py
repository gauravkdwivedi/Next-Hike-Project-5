# Import important libraries
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_excel('telcom_data.xlsx')
print(df)

# Check data information
df.info()

# Check null values
df.isnull().sum()

# Replace empty strings and spaces with NaN
df.replace({'': pd.NA, ' ': pd.NA}, inplace=True)

# Check null values again
df.isnull().sum()

# Calculate the percentage of missing values in the dataset
def percent_missing(df):

    # Calculate total number of cells in dataframe
    totalCells = np.prod(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

percent_missing(df)

# % of missing values in each column
missing_values = df.isnull()

# Calculate the percentage of missing values for each column
missing_percentage = (missing_values.sum() / len(df)) * 100

# Display the result
print("Percentage of missing values in each column:")
print(missing_percentage)

# Find categorical features
categorical = [X for X in df.columns if df[X].dtype=='object']

print('The categorical variables are: ',categorical)

# Assuming Bearer Id and MSISDN/Number are customer identification numbers. Since we have NaN values in these columns, I am removing lines with NaN values.

# Drop rows with NaN values for Bearer Id and MSISDN/Number
df = df.dropna(subset=['Bearer Id', 'MSISDN/Number'], axis=0)

# Checking null values
df.isnull().sum()

print(df)

# Export cleaned data
df.to_excel('telcom_data_cleaned.xlsx')