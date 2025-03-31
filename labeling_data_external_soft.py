import csv
import time
from pynput import keyboard
import threading

# Define the labels and their corresponding keys
labels = {
    '1': 'left_blink',
    '2': 'right_blink',
    '3': 'both_blink',
    '4': 'eyebrow_raise',
    '5': 'nothing'
}

# Initialize variables
current_label = 'nothing'  # Default label
start_time = time.time()
data = []  # List to store time and labels
running = True

# Function to update the current label based on keyboard input
def on_press(key):
    global current_label
    try:
        if key.char in labels:
            current_label = labels[key.char]
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Timer function to record time and labels
def record_data():
    global running
    while running:
        elapsed_time = time.time() - start_time
        data.append((elapsed_time, current_label))
        print(f"Time: {elapsed_time:.1f} s, Label: {current_label}")
        time.sleep(0.1)

# Function to save data to a CSV file
def save_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Label'])
        writer.writerows(data)

# Start the timer thread
thread = threading.Thread(target=record_data)
thread.start()

# Start listening for keyboard input
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        pass

# Stop the timer thread
running = False
thread.join()

# Save the data to a CSV file
save_to_csv('action_labels.csv')
print("Data saved to action_labels.csv")
