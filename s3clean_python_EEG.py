import pandas as pd

# gives digital filters
from scipy.signal import butter, filtfilt, iirnotch

"""
butter: butterworth band-pass filter: throw everything out around wanted frequency range
-gets noisy signals out, 0.5 - 50 contains the proper brain waves, boost these like an equalizer

include nyquist frequency to not mislabel fast waves as slow waves due to aliasing
aliasing: sampling illusion
"""

filename = 'combined_eeg_data_continuous.csv' 
combined_eeg = pd.read_csv(filename, header = None)

print("Data preview:")
print(combined_eeg.head())

# Assign proper column names, learned from OpenBCI 
combined_eeg.columns = [
    'Time', 'Marker', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 
    'Aux1', 'Aux2', 'Aux3', 'Aux4', 'Other1', 'Other2', 'Other3', 'Other4', 'Epoch', 'Other', 'Label'
]

# Print to verify corrected column alignment
print("\nCorrected Data Preview:")
print(combined_eeg.head())

"""
Currently using four electrodes:
Channel 1: below left eye
Channel 2: above left eye
Channel 3: below right eye
Channel 4: above right eye
"""

# Extract relevant columns: Time, Channels 1-4, Labels
relevant_data = combined_eeg[['Time', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 'Label']]

# Rename columns
relevant_data.columns = ['Time', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Label']

print("got relevant columns")

# Define the bandpass filter
"""
-Allows frequencies within a certain range to pass through
-Helps remove motion artifacts or EMG noise
-0.5-50 Hz
"""
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Define the notch filter
"""
-removes a certain frequency
-can be used to remove powerline interference
(not using for now)
"""
def notch_filter(data, notch_freq, fs, quality=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, quality)
    return filtfilt(b, a, data)

#Sampling rate
fs = 200  # OpenBCI Ganglion sampling rate

#Bandpass filter settings
lowcut = 0.5  # Low cutoff frequency (Hz)
highcut = 50  # High cutoff frequency (Hz)

# Notch filter settings
# notch_freq = 60  # Notch filter frequency (Hz)

relevant_data['Filtered Channel 1'] = bandpass_filter(relevant_data['Channel 1'], lowcut, highcut, fs)
relevant_data['Filtered Channel 2'] = bandpass_filter(relevant_data['Channel 2'], lowcut, highcut, fs)
relevant_data['Filtered Channel 3'] = bandpass_filter(relevant_data['Channel 3'], lowcut, highcut, fs)
relevant_data['Filtered Channel 4'] = bandpass_filter(relevant_data['Channel 4'], lowcut, highcut, fs)

# Save the filtered data to a new CSV file
output_filename = 'filtered_eeg_action_data.csv'
relevant_data.to_csv(output_filename, index=False)

print(f"Filtered data saved to '{output_filename}'")
