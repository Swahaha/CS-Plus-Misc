import pandas as pd

# Path to your pickle file
pkl_file = 'dataa/feature_matching/data.pkl'

# Read the pickle file
data = pd.read_pickle(pkl_file)

# Display the structure of the DataFrame
print("DataFrame structure:")
print(data.info())

# Display the first few rows of the DataFrame
#print("\nFirst few rows of the DataFrame:")
#print(data.head())

# Display the column names
print("\nColumn names:")
print(data.columns)

# Display the content of the 'matchings' column for the first row to understand its structure
# print("\nContent of the 'matchings' column for the first row:")
# print(data.iloc[0]['matchings'])
