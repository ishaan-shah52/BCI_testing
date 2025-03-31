import matplotlib.pyplot as plt
import pandas as pd

# Load EEG data from the CSV file
filename = 'filtered_eeg_action_data.csv'
data = pd.read_csv(filename)

# Debug: Print the first few rows of the dataset
print("Dataset preview:")
print(data.head())

# Assuming the CSV columns are as follows:
# 0: Time (s)
# 1: Channel 2
# 2: Channel 4
# 3: Label
# 4: Filtered Channel 2
# 5: Filtered Channel 4

# Extract time, labels, and the filtered EEG channels
time = data.iloc[:, 0]           # Time (s)
labels = data.iloc[:, 3]         # Label
channels = data.iloc[:, [4, 5]]    # Filtered Channel 2 and Filtered Channel 4

# Rename channels for clarity (optional)
channels.columns = ['Filtered Channel 2', 'Filtered Channel 4']

# Define a mapping from label to a color
label_colors = {
    'nothing': 'gray',
    'left_blink': 'blue',
    'right_blink': 'red',
    'both_blink': 'purple',
    'eyebrow_raise': 'green'
}

# Create a copy of the data for segmenting labels
df = data.copy()
df['Time'] = time
df['Label'] = labels

# Identify segments where the label is constant.
# A new segment starts when the label changes compared to the previous row.
df['Label_Change'] = df['Label'] != df['Label'].shift(1)
df['Segment'] = df['Label_Change'].cumsum()

# Create subplots: one for the EEG data and one for the label timeline.
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                               gridspec_kw={'height_ratios': [4, 1]})

# Plot the EEG channels on the first subplot.
ax1.plot(time, channels['Filtered Channel 2'], label='Filtered Channel 2')
ax1.plot(time, channels['Filtered Channel 4'], label='Filtered Channel 4')
ax1.set_ylim(-20000, 20000)
ax1.set_title('EEG Channel Data')
ax1.set_ylabel('Amplitude (µV)')
ax1.legend()
ax1.grid(True)

# Plot the label timeline on the second subplot.
# For each continuous segment with the same label, fill that time range with a color.
segments = df.groupby('Segment')
for seg, group in segments:
    seg_label = group['Label'].iloc[0]
    start_time = group['Time'].iloc[0]
    end_time = group['Time'].iloc[-1]
    color = label_colors.get(seg_label, 'black')  # Default to black if label not found

    # Fill the background for this label segment.
    ax2.axvspan(start_time, end_time, color=color, alpha=0.5)
    
    # Optionally, place the label text in the middle of the segment.
    mid_time = (start_time + end_time) / 2
    ax2.text(mid_time, 0.5, seg_label,
             horizontalalignment='center', verticalalignment='center',
             fontsize=10, color='white', transform=ax2.get_xaxis_transform())

ax2.set_yticks([])  # Hide y-axis ticks on the label timeline.
ax2.set_xlabel('Time (s)')
ax2.set_title('Label Timeline')

plt.tight_layout()
plt.show()
