import pandas as pd
import glob

#grab all csv files with data
#glob library uses wildcard naming convention with *
file_pattern = "eeg_sessions/eeg_action_data_*.csv"
file_list = sorted(glob.glob(file_pattern))
print("Found files:", file_list)

eachEEG = []
time_offset = 0.0

for file in file_list:
    #Read CSV without headers
    currentEEG = pd.read_csv(file, header=None)
    
    #make first column time
    currentEEG[0] = pd.to_numeric(currentEEG[0], errors='coerce')
    
    # Add the current offset to the time column
    currentEEG[0] += time_offset
    
    # Update the offset: get the maximum time value in this session for the next starting number for no overlap
    session_max_time = currentEEG[0].max()
    time_offset = session_max_time
    
    eachEEG.append(currentEEG)

# Check that dataframes is not empty
if not eachEEG:
    raise ValueError("No files were found or no data was read.")

# Concatenate all DataFrames into one continuous DataFrame, ignores the index when concatenating
allEEG = pd.concat(eachEEG, ignore_index=True)

# Optionally, sort by time (should be already in order)
allEEG.sort_values(by=0, inplace=True)

# Save the combined DataFrame to a new CSV file without headers (if desired)
allEEG.to_csv("combined_eeg_data_continuous.csv", index=False, header=False)

print("All sessions have been combined into 'combined_eeg_data_continuous.csv' with continuous time.")
