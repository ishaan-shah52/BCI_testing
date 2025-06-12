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

current_label = 'nothing'
start_time = time.time()
label_data = []  #for storing time and actions
eeg_data = []  #to store voltages from EEG
running = True

#BrainFlow setup based on documentation: https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#ganglion
params = BrainFlowInputParams()
params.mac_address = "D5:A4:BE:DD:BC:89"  #this was recieved from using python libraries in the previous step file (s0) so device is instantly found
params.serial_port = "COM5"  
board = BoardShim(BoardIds.GANGLION_BOARD.value, params)

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
        elapsed_time = time.time() - start_time
        label_data.append((elapsed_time, current_label))
        print(f"Time: {elapsed_time:.1f} s, Label: {current_label}") #.1f is a floating point to one decimal place
        time.sleep(0.1) #record every 0.1 seconds

#record EEG data
def record_eeg():
    global running
    while running:
        elapsed_time = time.time() - start_time
        eeg_samples = board.get_current_board_data(1)  # Get the latest EEG sample
        if eeg_samples.shape[1] > 0:  # Ensure data is available
            eeg_sample = eeg_samples[:, -1]  # Latest sample
            eeg_data.append((elapsed_time, *eeg_sample))
        time.sleep(0.1)  # Match label sampling interval

# Merge EEG data with labels based on timestamps
def merge_data():
    merged_data = []
    for eeg_sample in eeg_data:
        eeg_time = eeg_sample[0]
        # Find the nearest label based on time
        closest_label = min(label_data, key=lambda x: abs(x[0] - eeg_time)) #not all times are lined up cuz of time.time
        merged_data.append((*eeg_sample, closest_label[1]))
    return merged_data

#CSV file
def save_to_csv(filename, merged_data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Time (s)', 'EEG_Ch1', 'EEG_Ch2', 'EEG_Ch3', 'EEG_Ch4', 'Label'])
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
        save_to_csv('eeg_sessions/eeg_action_data_13.csv', merged_data) #change both of these lines
        print("Data saved to 'eeg_action_data_13.csv'") #this one

        print("Releasing session...")
        board.release_session()
        print("Session released.")

if __name__ == "__main__":
    main()
