# Import important libraries
import pandas as pd

# Load the cleaned dataset
df = pd.read_excel('telcom_data_cleaned.xlsx')

print(df)

# Create user engagement data
df['Session_Frequency'] = df.groupby(by=['MSISDN/Number'])['Dur. (ms)'].transform('count')
df['Session_Duration'] = df['Dur. (ms)']
df['Total_Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

print(df)

user_engagement_data = df[['Bearer Id', 'MSISDN/Number', 'Session_Frequency', 'Session_Duration', 'Social Media DL (Bytes)',
                           'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 
                           'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 
                           'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Total_Traffic']]

print(user_engagement_data)

user_engagement_data = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'sum',
    'Session_Frequency': 'sum',
    'Session_Duration' : 'sum',
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
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Total_Traffic': 'sum'
}).reset_index()

print(user_engagement_data)

# Export user_engagement_data
user_engagement_data.to_excel('user_engagement_data.xlsx', index=False)