import pandas as pd
import glob

# Adjust the pattern to match your CSV files
file_pattern = "eeg_action_data_*.csv"
file_list = sorted(glob.glob(file_pattern))
print("Found files:", file_list)

dataframes = []
time_offset = 0.0

for file in file_list:
    # Read CSV without headers
    df = pd.read_csv(file, header=None)
    
    # Ensure the first column (time) is numeric
    df[0] = pd.to_numeric(df[0], errors='coerce')
    
    # Add the current offset to the time column
    df[0] += time_offset
    
    # Update the offset: get the maximum time value in this session
    session_max_time = df[0].max()
    time_offset = session_max_time
    
    dataframes.append(df)

# Check that dataframes is not empty
if not dataframes:
    raise ValueError("No files were found or no data was read.")

# Concatenate all DataFrames into one continuous DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Optionally, sort by time (should be already in order)
combined_df.sort_values(by=0, inplace=True)

# Save the combined DataFrame to a new CSV file without headers (if desired)
combined_df.to_csv("combined_eeg_data_continuous.csv", index=False, header=False)

print("All sessions have been combined into 'combined_eeg_data_continuous.csv' with continuous time.")
