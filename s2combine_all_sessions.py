
"""
Notes:
Merges all recorded EEGs into one continuous dataset
-adds session ID to know where it came from

This step is important for Machine learning training from one file
and to have a unified dataset to get statistics and point estimates from
and one path for handling all of the data preprocessing and filtering in the next step. 

Libraries:
glob is for finding file names with similar names
-using wildcard pattern, sorted unncessary
"""
import pandas as pd
import glob, os

#grab all csv files with data
#glob library uses wildcard naming convention with *
file_pattern = "eeg_sessions/eeg_action_data_*.csv"
file_list = sorted(glob.glob(file_pattern)) #unnecessary to sort
print("Found files:", file_list)

eachEEG = []

for idx, file in enumerate(file_list):
    EEG_file = pd.read_csv(file, header=0)
    EEG_file["session_id"] = idx
    eachEEG.append(EEG_file) #array of all files

if not eachEEG:
    raise ValueError("No files found.")

#need ignore_index to make sure indices from files do not overlap and rewrite each other in the file
allEEG = pd.concat(eachEEG, ignore_index=True) #puts all files tg
allEEG.to_csv("combined_eeg_data.csv", index=False)
print("Combined -> combined_eeg_data.csv")

print("Final shape:", allEEG.shape)
