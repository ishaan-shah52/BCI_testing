import time
import numpy as np
import tensorflow as tf
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Load your pre-trained CNN model
model = tf.keras.models.load_model('EEG_CNN_model.h5')

# Define your label mapping if needed (same as used in training)
label_map = {0: 'nothing', 1: 'left_blink', 2: 'right_blink', 3: 'both_blink', 4: 'eyebrow_raise'}

# Set up BrainFlow parameters (adjust these as needed for your board)
params = BrainFlowInputParams()
params.serial_port = "COM5"          # or adjust for your system
params.mac_address = "D5:A4:BE:DD:BC:89"  # if using a Ganglion board with BLE, for example
board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
board.prepare_session()
board.start_stream()

# Define epoch parameters
epoch_length = 3.0  # seconds
fs = board.get_sampling_rate(BoardIds.GANGLION_BOARD.value)
samples_per_epoch = int(epoch_length * fs)

# Define your preprocessing function (this should mimic your training preprocessing)
def preprocess(epoch_data):
    """
    epoch_data: numpy array of shape (samples_per_epoch, num_channels)
    Apply any filtering, scaling, normalization etc. here.
    For now, assume data is already filtered.
    """
    # Example: normalize each channel to zero mean and unit variance
    epoch_data = (epoch_data - np.mean(epoch_data, axis=0)) / np.std(epoch_data, axis=0)
    return epoch_data

print("Starting live classification. Press Ctrl+C to stop.")

try:
    while True:
        # Get the latest data for one epoch from the board
        # This returns an array with shape (num_channels, num_samples)
        data = board.get_current_board_data(samples_per_epoch)
        
        # Select the channels you want to use (for example, filtered EEG channels)
        # Adjust channel indices as needed. Here, we're assuming channels 4 and 5 are your filtered channels.
        # (Indices may differ depending on how your CSV was structured.)
        selected_channels = [4, 5]
        epoch_data = data[selected_channels, :].T  # Now shape: (samples_per_epoch, num_channels)
        
        # Preprocess the data to match the model's expected input
        epoch_data = preprocess(epoch_data)
        # Suppose min_samples was determined during training

        min_samples = 12  # use the same value from your training code

        if epoch_data.shape[0] > min_samples:
            epoch_data = epoch_data[:min_samples, :]
        elif epoch_data.shape[0] < min_samples:
            # Optionally, pad the epoch_data to reach min_samples
            pad_width = min_samples - epoch_data.shape[0]
            epoch_data = np.pad(epoch_data, ((0, pad_width), (0, 0)), mode='constant')

        # Expand dims to add the batch dimension: (1, samples_per_epoch, num_channels)
        epoch_data = np.expand_dims(epoch_data, axis=0)
        
        # Run the model prediction on this epoch
        prediction = model.predict(epoch_data)
        predicted_class = np.argmax(prediction)
        predicted_label = label_map[predicted_class]
        
        # Print or use the prediction
        print("Predicted label:", predicted_label)
        
        # Wait for the next epoch (or use a sliding window approach for more frequent updates)
        time.sleep(epoch_length)
        
except KeyboardInterrupt:
    print("Live classification stopped.")

finally:
    board.stop_stream()
    board.release_session()
