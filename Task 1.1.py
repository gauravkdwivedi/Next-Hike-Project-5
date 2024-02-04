# Import important libraries
import numpy as np
import pandas as pd

# Load the cleaned dataset
df = pd.read_excel('C:/Users/gaura/OneDrive/Visual Studio/NextHike Project 5/telcom_data_cleaned.xlsx')

# Create new dataset for user behavior
user_behavior_data = df[['Bearer Id', 'MSISDN/Number', 'Start ms', 'End ms', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 
                       'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 
                       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                       'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']]
print(user_behavior_data)

# Check the description of the new dataset
user_behavior_data.describe().round(2)

# Aggregate per user the following information
# number of xDR sessions
# Session duration
# the total download (DL) and upload (UL) data
# the total data volume (in Bytes) during this session for each application

user_behavior_data['number of xDR sessions'] = user_behavior_data['Bearer Id']

user_behavior_data = user_behavior_data.groupby('MSISDN/Number').agg({
    'number of xDR sessions': 'count',
    'Start ms': 'min',
    'End ms': 'max',
    'Dur. (ms)': 'sum',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Email UL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Youtube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Gaming UL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum',
    'Other UL (Bytes)': 'sum',
}).reset_index()

print(user_behavior_data)

# Export users_behavior_data
user_behavior_data.to_excel('C:/Users/gaura/OneDrive/Visual Studio/NextHike Project 5/users_behavior_data.xlsx')