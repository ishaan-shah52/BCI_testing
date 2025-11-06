
"""
Notes:
loads the combined csv with the eeg data
-sorts by time and session
-filters with butterworth band pass from 0.5-20 hz
 since this is where blinks and eyebrow activity is
-then applies zero-phase to cancel phase distortion
-downsamples

improvements:
make sure that downsample does not cut between sessions
"""
import pandas as pd
import numpy as np
# gives digital filters
from scipy.signal import butter, filtfilt, iirnotch

"""
butter: butterworth band-pass filter: throw everything out around wanted frequency range
-gets noisy signals out, 0.5 - 50 contains the proper brain waves, boost these like an equalizer

filtfilt: we want features to line up with blink events, for offline training, goes forward and backward for zero phase shift
-not using sosfilt here because of lag, only can do forward pass, which introduces a lag

include nyquist frequency to not mislabel fast waves as slow waves due to aliasing
-sample at least 2 times the fastest wanted which is 50, so 100 is slowest to sample at
due to two vertices of full wave rule
aliasing: sampling illusion
"""

filename = 'combined_eeg_data.csv' 

#research for eye signal based on: https://www.researchgate.net/figure/The-frequency-response-of-eye-blinking-signal_fig4_275830679

FS = 200 # Ganglion sampling rate
LOWCUT = 0.5       
HIGHCUT = 20.0 # EEG mostly < 20 Hz
ORDER = 4
DOWNSAMPLE_TO = 50
APPLY_NOTCH_60 = False  
CHANNELS = ['EEG_Ch1','EEG_Ch2','EEG_Ch3','EEG_Ch4']

combined_eeg = pd.read_csv(filename)

print("Data preview:")
print(combined_eeg.head())

#quick check
required_cols = ['BoardTime', *CHANNELS, 'Label']
missing = [c for c in required_cols if c not in combined_eeg.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

#make sure values are numeric
for col in ['BoardTime', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4']:
    combined_eeg[col] = pd.to_numeric(combined_eeg[col], errors='coerce')

# Drop rows with missing required fields
combined_eeg.dropna(subset=required_cols, inplace=True)

# Detect optional metadata columns that might exist if you added them during combine
has_session = 'session_id' in combined_eeg.columns

#FILTERING

# Define the bandpass filter
"""
-Allows frequencies within a certain range to pass through
-Helps remove motion artifacts or EMG noise
-0.5-50 Hz
"""

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq) #for stability
    high = min(0.999999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

#block as parameter to do by session rather than all at once
def apply_filters(block, fs):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, ORDER)
    for ch in CHANNELS:
        #this is acausal: peeks into future
        block[ch] = filtfilt(b, a, block[ch].to_numpy(), method='gust')
    return block

if has_session:
    combined_eeg = combined_eeg.sort_values(['session_id', 'BoardTime'])
    combined_eeg = combined_eeg.groupby('session_id', group_keys=False).apply(lambda g: apply_filters(g, FS))
else:
    # just apply once to all data
    combined_eeg = combined_eeg.sort_values('BoardTime')
    combined_eeg = apply_filters(combined_eeg, FS)

#could down sample since highest freq needed is 20 Hz and nyquist says you only need 2x
#reduces file size
if DOWNSAMPLE_TO and DOWNSAMPLE_TO < FS and FS % DOWNSAMPLE_TO == 0:
    step = FS // DOWNSAMPLE_TO #200/50 = 4
    if 'session_id' in combined_eeg.columns:
        combined_eeg = combined_eeg.groupby('session_id', group_keys=False).apply(lambda g: g.iloc[::step].copy())
    else:
        combined_eeg = combined_eeg.iloc[::step].copy()

print("Combined filtered EEG shape:", combined_eeg.shape)
# Save the filtered data to a new CSV file
output_filename = 'filtered_eeg_action_data.csv'
combined_eeg.to_csv(output_filename, index=False)

print(f"Filtered data saved to '{output_filename}'")
