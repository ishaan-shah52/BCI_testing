import csv
import time
import threading
from pynput import keyboard
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

#labels and associated numbers on keyboard
labels = {
    '1': 'left_blink',
    '2': 'right_blink',
    '3': 'both_blink',
    '4': 'eyebrow_raise',
    '5': 'nothing'
}

"""
Recording set up:
Channel 1: above left eye
Channel 2: below left eye
Channel 3: above right eye
Channel 4: below right eye
"""

current_label = 'nothing'
# start_time = time.time()
label_data = []  #for storing time and actions
eeg_data = []  #to store voltages from EEG
running = True

#BrainFlow setup based on documentation: https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#ganglion
params = BrainFlowInputParams()
params.mac_address = "D5:A4:BE:DD:BC:89"  #this was recieved from using python libraries in the previous step file (s0) so device is instantly found
params.serial_port = "COM5"  
board = BoardShim(BoardIds.GANGLION_BOARD.value, params)

eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)

# Function to update the current label based on keyboard input
def on_press(key):
    global current_label
    try:
        if key.char in labels:
            current_label = labels[key.char]
    except AttributeError:
        pass

#Stop on escape
def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

#Timer to also record labels
def record_labels():
    global running 
    while running:
        elapsed_time = time.time()
        label_data.append((elapsed_time, current_label))
        print(f"Time: {elapsed_time:.1f} s, Label: {current_label}") #.1f is a floating point to one decimal place
        time.sleep(0.1) #record every 0.1 seconds

#record EEG data
def record_eeg():
    global running
    while running:
        eeg_samples = board.get_board_data()  # Get the all EEG samples unprocessed
        if eeg_samples.shape[1] > 0:  # Ensure data is available
            for i in range(eeg_samples.shape[1]):
                timestamp = eeg_samples[-1, i]  # Board timestamp
                ch_values = [eeg_samples[ch, i] for ch in eeg_channels]
                eeg_data.append((timestamp, *ch_values))
        time.sleep(0.05)  

# Merge EEG data with labels based on timestamps
def merge_data():
    merged = []
    if not eeg_data or not label_data:
        return merged

    # Align wall clock labels with board timestamps
    first_board_time = eeg_data[0][0]
    first_wall_time = label_data[0][0]
    time_offset = first_wall_time - first_board_time  # difference in seconds

    for sample in eeg_data:
        board_time = sample[0]
        # Convert board time to wall time
        wall_time_est = board_time + time_offset
        # Find nearest label
        closest_label = min(label_data, key=lambda x: abs(x[0] - wall_time_est))
        merged.append((*sample, closest_label[1]))
    return merged

#CSV file
def save_to_csv(filename, merged_data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['BoardTime', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 'Label'])
        writer.writerows(merged_data)

# Main function to start EEG and label recording
def main():
    global running
    try:
        print("Preparing session...")
        board.prepare_session()

        print("Starting EEG data stream...")
        board.start_stream()

        #use threading for least delay between collection of data
        label_thread = threading.Thread(target=record_labels)
        eeg_thread = threading.Thread(target=record_eeg)
        label_thread.start()
        eeg_thread.start()

        # Start listening for keyboard input in a non-blocking way
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("Press keys to label actions (1: left_blink, 2: right_blink, etc.)")
        print("Press ESC to stop recording, or Ctrl+C to force exit.")

        # Keep the main thread alive, but allow KeyboardInterrupt to be caught
        while listener.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Stopping data collection...")
    finally:
        # Ensure threads and board stream are stopped
        running = False
        listener.stop()
        label_thread.join()
        eeg_thread.join()

        print("Stopping EEG data stream...")
        board.stop_stream()

        # Merge data and save to CSV
        print("Merging EEG data with labels...")
        merged_data = merge_data()
        save_to_csv('eeg_sessions/eeg_action_data_1.csv', merged_data) #change both of these lines
        print("Data saved to 'eeg_action_data_1.csv'") #this one
        print("change these file numbers now")

        print("Releasing session...")
        board.release_session()
        print("Session released.")

if __name__ == "__main__":
    main()
