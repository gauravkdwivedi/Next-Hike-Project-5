# Import important libraries
import numpy as np
import pandas as pd

# Load the cleaned dataset
df = pd.read_excel('telcom_data_cleaned.xlsx')

# Find the top 10 'Handset Type'
df['Handset Type'].value_counts().head(10)

# Find the top 3 'Handset Manufacturer'
df['Handset Manufacturer'].value_counts().head(3)

# Top 5 handsets for the top 3 handset manufacturers
top_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index

for manufacturer in top_manufacturers:
    # Filter DataFrame for rows with the current manufacturer
    manufacturer_df = df[df['Handset Manufacturer'] == manufacturer]

    # Get the top 5 handsets for the current manufacturer
    top_handsets = manufacturer_df['Handset Type'].value_counts().head(5)

    # Display the results
    print(f"\nTop 5 handsets for {manufacturer}:\n{top_handsets}")

# After conducting an analysis, it has been determined that the leading handset manufacturers are Apple, Samsung, and Huawei, with the respective top 5 handsets listed above.