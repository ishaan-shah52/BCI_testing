import pandas as pd
import glob

######################################
# Step 1: Combine and Normalize Files
######################################

# Adjust the pattern to match your CSV files
file_pattern = "eeg_action_data_*.csv"
file_list = sorted(glob.glob(file_pattern))
print("Found files:", file_list)

if not file_list:
    raise ValueError("No files found matching the pattern.")

dataframes = []
time_offset = 0.0

# In the combined data, we assume the columns are:
# Index 0: Time, 1: Marker, 2: EEG_Ch1, 3: EEG_Ch2, 4: EEG_Ch3, 5: EEG_Ch4, ...
# We'll normalize EEG channels (columns 2-5) across sessions.
channels_to_normalize = [2, 3, 4, 5]

# --- Process the baseline file (first file) ---
baseline_df = pd.read_csv(file_list[0], header=None)
baseline_df[0] = pd.to_numeric(baseline_df[0], errors='coerce')
baseline_df[0] += time_offset
time_offset = baseline_df[0].max()

# Compute baseline statistics for the EEG channels
baseline_means = baseline_df[channels_to_normalize].mean()
baseline_stds = baseline_df[channels_to_normalize].std()

# Append the baseline (no change needed)
dataframes.append(baseline_df)

# --- Process subsequent files (new sessions) ---
for file in file_list[1:]:
    df = pd.read_csv(file, header=None)
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df[0] += time_offset
    time_offset = df[0].max()
    
    # Compute current file statistics for EEG channels
    new_means = df[channels_to_normalize].mean()
    new_stds = df[channels_to_normalize].std()
    
    # Normalize each EEG channel so that its distribution matches the baseline
    for col in channels_to_normalize:
        df[col] = ((df[col] - new_means[col]) / new_stds[col]) * baseline_stds[col] + baseline_means[col]
    
    dataframes.append(df)

# Concatenate all sessions into one continuous DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.sort_values(by=0, inplace=True)

# Save combined data (without headers) to a CSV file
combined_output_filename = "combined_eeg_data_continuous.csv"
combined_df.to_csv(combined_output_filename, index=False, header=False)
print(f"Combined normalized data saved to '{combined_output_filename}'")
