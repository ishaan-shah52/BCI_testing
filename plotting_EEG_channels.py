import matplotlib.pyplot as plt
import pandas as pd

# Load EEG data from the CSV file
filename = 'filtered_eeg_action_data.csv'
data = pd.read_csv(filename)

# Debug: Print the first few rows of the dataset
print("Dataset preview:")
print(data.head())

# Extract time and EEG channels
time = data.iloc[:, 0]  # Assuming the first column is time
print("\nTime column preview:")
print(time.head())

# Adjust these indices to match the EEG channel columns
unfiltered_channel_indices = [1, 2]  # Update as per your dataset
filtered_channel_indices = [4, 5]  # Update as per your dataset
all_channel_indices = [1, 2, 4, 5]  # Update as per your dataset

channels = data.iloc[:, filtered_channel_indices]
print("\nEEG Channels preview:")
print(channels.head())

# Ensure all values are finite
if not channels.applymap(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).all().all():
    raise ValueError("Data contains non-numeric or NaN values.")

# Plot each EEG channel
plt.figure(figsize=(12, 8))
for i, col in enumerate(channels.columns):
    plt.plot(time, channels[col], label=f'EEG_Ch{i+1}')

# Set y-axis range
plt.ylim(-20000, 20000)

# Add labels, legend, and grid
plt.title('EEG Channel Data')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
