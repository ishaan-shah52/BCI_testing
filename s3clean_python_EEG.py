import pandas as pd
import numpy as np
# gives digital filters
from scipy.signal import butter, filtfilt, iirnotch

"""
butter: butterworth band-pass filter: throw everything out around wanted frequency range
-gets noisy signals out, 0.5 - 50 contains the proper brain waves, boost these like an equalizer

include nyquist frequency to not mislabel fast waves as slow waves due to aliasing
-sample at least 2 times the fastest wanted which is 50, so 100 is slowest to sample at
due to two vertices of full wave rule
aliasing: sampling illusion
"""

filename = 'combined_eeg_data.csv' 
combined_eeg = pd.read_csv(filename, header = None)

print("Data preview:")
print(combined_eeg.head())

FS = 200           # Ganglion sampling rate
LOWCUT = 0.5       # blink/eyebrow energy starts ~0.5 Hz
HIGHCUT = 20.0     # EEG mostly < 20 Hz
ORDER = 4
DOWNSAMPLE_TO = 50 # set to None to skip; else 50 is good

rename_map = {0:'BoardTime', 1:'EEG_Ch1', 2:'EEG_Ch2', 3:'EEG_Ch3', 4:'EEG_Ch4', 5:'Label'}
combined_eeg.rename(columns=rename_map, inplace=True)

for col in ['BoardTime', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4']:
    combined_eeg[col] = pd.to_numeric(combined_eeg[col], errors='coerce')

# Drop rows with missing required fields
combined_eeg.dropna(subset=['BoardTime', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 'Label'], inplace=True)

# Detect optional metadata columns that might exist if you added them during combine
has_session = 'session_id' in combined_eeg.columns
has_file = 'file' in combined_eeg.columns

#FILTERING

# Define the bandpass filter
"""
-Allows frequencies within a certain range to pass through
-Helps remove motion artifacts or EMG noise
-0.5-50 Hz
"""

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(LOWCUT, HIGHCUT, FS, ORDER)
channels = ['EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4']

# If you kept session boundaries, filter per session to avoid edge artifacts across sessions
if has_session:
    def filt_group(g):
        g = g.sort_values('BoardTime')
        for ch in channels:
            # filtfilt is zero-phase (great offline), requires enough samples
            if len(g[ch]) > 3 * max(len(a), len(b)):
                g[ch] = filtfilt(b, a, g[ch].to_numpy(), method='gust')
            else:
                # fallback: skip filtering for tiny groups
                pass
        return g
    combined_eeg = combined_eeg.groupby('session_id', group_keys=False).apply(filt_group)
else:
    combined_eeg = combined_eeg.sort_values('BoardTime')
    for ch in channels:
        if len(combined_eeg[ch]) > 3 * max(len(a), len(b)):
            combined_eeg[ch] = filtfilt(b, a, combined_eeg[ch].to_numpy(), method='gust')

# -----------------------------
# OPTIONAL DOWNSAMPLE (after LP filtering)
# Simple stride downsample is OK here because HIGHCUT<=15 Hz and FS→50 Hz.
# -----------------------------
if DOWNSAMPLE_TO is not None and DOWNSAMPLE_TO < FS and FS % DOWNSAMPLE_TO == 0:
    step = FS // DOWNSAMPLE_TO

    def downsample_block(g):
        return g.iloc[::step].copy()

    if has_session:
        combined_eeg = combined_eeg.groupby('session_id', group_keys=False).apply(downsample_block)
    else:
        combined_eeg = combined_eeg.iloc[::step].copy()

# Save the filtered data to a new CSV file
output_filename = 'filtered_eeg_action_data.csv'
combined_eeg.to_csv(output_filename, index=False)

print(f"Filtered data saved to '{output_filename}'")
