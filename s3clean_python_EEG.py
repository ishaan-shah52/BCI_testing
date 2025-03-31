import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

# Load the CSV file without assuming headers
filename = 'combined_eeg_data_continuous.csv'  # Replace with your actual file name
data = pd.read_csv(filename, header = None)  # For tab-separated values

# Print the first few rows for inspection
print("Raw data preview:")
print(data.head())

# Assign proper column names
data.columns = [
    'Time', 'Marker', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 
    'Aux1', 'Aux2', 'Aux3', 'Aux4', 'Other1', 'Other2', 'Other3', 'Other4', 'Epoch', 'Other', 'Label'
]

# Print to verify corrected column alignment
print("\nCorrected Data Preview:")
print(data.head())

# Extract relevant columns: Time, Channel 2, Channel 4, and Label
relevant_data = data[['Time', 'EEG_Ch2', 'EEG_Ch4', 'Label']]

# Rename columns for clarity
relevant_data.columns = ['Time (s)', 'Channel 2', 'Channel 4', 'Label']

print("here")

# Define the bandpass filter
"""
Allows frequencies within a certain range to pass through

Helps remove motion artifacts or EMG noise

0.5-50 Hz
"""
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Define the notch filter
"""
removes a certain frequency

can be used to remove powerline interference
"""
def notch_filter(data, notch_freq, fs, quality=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, quality)
    return filtfilt(b, a, data)

# Sampling rate (in Hz)
fs = 200  # OpenBCI Ganglion sampling rate

# Bandpass filter settings
lowcut = 0.5  # Low cutoff frequency (Hz)
highcut = 50  # High cutoff frequency (Hz)

# Notch filter settings
# notch_freq = 60  # Notch filter frequency (Hz)

relevant_data['Filtered Channel 2'] = bandpass_filter(relevant_data['Channel 2'], lowcut, highcut, fs)
relevant_data['Filtered Channel 4'] = bandpass_filter(relevant_data['Channel 4'], lowcut, highcut, fs)

# Save the filtered data to a new CSV file
output_filename = 'filtered_eeg_action_data.csv'
relevant_data.to_csv(output_filename, index=False)

print(f"Filtered data saved to '{output_filename}'")
