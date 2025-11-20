import pandas as pd

# Read the CSV file
df = pd.read_csv("combined_sorted.csv")

# Filter rows where dihedral is between 50 and 60
filtered_df = df[(df['dihedral'] >= 50) & (df['dihedral'] <= 60)]

# Show statistics for rz and energy columns in the filtered data
print("Statistics for rz where dihedral is between 50 and 60:")
print(filtered_df['rz'].describe())
print("\nStatistics for energy where dihedral is between 50 and 60:")
print(filtered_df['energy'].describe())

# Print the rows that match the criteria
print("\nRows with dihedral angles between 50 and 60:")
print(filtered_df)
