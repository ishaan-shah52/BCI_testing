import matplotlib.pyplot as plt
import pandas as pd

#post filtering
filename = 'filtered_eeg_action_data.csv'
filtered_eeg = pd.read_csv(filename)

#quick check
print("Dataset preview:")
print(filtered_eeg.head())

# Get time, labels, and the filtered EEG channels columns
time = filtered_eeg['Time']
labels = filtered_eeg['Label']

filtered_cols = [
    'Filtered Channel 1',
    'Filtered Channel 2',
    'Filtered Channel 3',
    'Filtered Channel 4'
]

channels = filtered_eeg[filtered_cols]

#corresponds to colored wires I am using
channel_colors = {
    'Filtered Channel 1': 'green',
    'Filtered Channel 2': 'yellow',
    'Filtered Channel 3': 'orange',
    'Filtered Channel 4': 'red'
}

#mapping from label to a color
label_colors = {
    'nothing': 'gray',
    'left_blink': 'blue',
    'right_blink': 'red',
    'both_blink': 'purple',
    'eyebrow_raise': 'green'
}

#Create a copy of the data for segmenting labels
# essentially this is a helper dataframe
helper_df = filtered_eeg.copy()

# Identify segments where the label is constant.
# A new segment starts when the label changes compared to the previous row.

#shift(1) shifts the whole row down by one with a NaN in the first row
helper_df['Label_Change'] = helper_df['Label'] != helper_df['Label'].shift(1) #true or false column
helper_df['Segment'] = helper_df['Label_Change'].cumsum() #keep a running count of True

# Create subplots: one for the EEG data and one for the label timeline.
fig, (ax1, ax2) = plt.subplots(
    2, 1,                     # 2 rows, 1 column
    sharex=True,              # both rows share the same x-axis
    figsize=(12, 8),          # width x height in inches
    gridspec_kw={'height_ratios': [4, 1]}  # top panel 4× taller than bottom
)

# Plot the EEG channels on the first subplot.
for col in filtered_cols: #4 channels
    ax1.plot(time,
             channels[col],
             label=col,
             color=channel_colors.get(col, 'black'))

ax1.set_ylabel('Amplitude (µV)')
ax1.set_title('Filtered EEG: Channels 1-4')
ax1.set_ylim(-5000, 5000)  
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot the label timeline on the second subplot.
# For each continuous segment with the same label, fill that time range with a color.
segments = helper_df.groupby('Segment')
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
