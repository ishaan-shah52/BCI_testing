import pandas as pd

# Load the tab-separated file
data = pd.read_csv("Recordings/BrainFlow-RAW_Recordings_1.csv", header=None, sep="\t")

# Inspect the first few rows to understand the structure
print(data.head())

# # Define the sample rate (e.g., 200 samples/second)
# sample_rate = 200  # Replace with your actual sample rate

# # Add a Time column based on the sample rate
# data['Time'] = data.index / sample_rate

# # Filter the first 10 seconds
# filtered_data = data[data['Time'] <= 10]

# Select the desired columns (adjust the column indices as needed)
# Column indices in pandas are 0-based, so 0 = first column, 2 = third column, 4 = fifth column
selected_columns = data.iloc[:, [0, 2, 4]]

# Rename the columns
selected_columns.columns = ['Time', 'Channel 2', 'Channel 4']

# Save the selected columns into a new CSV file
output_file = "neural_activity.csv"  # Replace with your desired output file name
selected_columns.to_csv(output_file, index=False)

print(f"Filtered CSV saved as {output_file}")

