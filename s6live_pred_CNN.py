import time
import numpy as np
import tensorflow as tf
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

#model
model = tf.keras.models.load_model('EEG_CNN_model.h5')

#label mapping
label_map = {0: 'nothing', 1: 'left_blink', 2: 'right_blink', 3: 'both_blink', 4: 'eyebrow_raise'}

# Set up BrainFlow parameters
params = BrainFlowInputParams()
params.serial_port = "COM5"      
params.mac_address = "D5:A4:BE:DD:BC:89" 
board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
board.prepare_session()
board.start_stream()

#epoch parameters
epoch_length = 2.0  # seconds
fs = board.get_sampling_rate(BoardIds.GANGLION_BOARD.value)
samples_per_epoch = int(epoch_length * fs)
min_samples = 19

#get eeg channels
eeg_channels = board.get_eeg_channels(BoardIds.GANGLION_BOARD.value)
print("Recording EEG from rows:", eeg_channels) 

def preprocess(epoch):
    #band-pass filter 0.5–50Hz
    for ch in range(epoch.shape[1]):
        DataFilter.perform_bandpass(
            epoch[:, ch], fs,
            25,    # center freq
            49.5,  # half-bandwidth = 50–0.5
            4, FilterTypes.BUTTERWORTH.value, 0
        )
    #Normalize each channel to zero mean, unit variance
    return (epoch - np.mean(epoch, axis=0)) / np.std(epoch, axis=0)


print("Starting live classification. Press Ctrl+C to stop.")

try:
    while True:
        # array with shape (num_channels, num_samples)
        raw = board.get_current_board_data(samples_per_epoch)
        
        #get enough data for preprocessing
        if raw.shape[1] < samples_per_epoch:
            continue  # not enough data yet

        #Slice out only the EEG rows = (samples, 4)
        epoch = raw[[2, 3, 4, 5], :].T

        
        # Preprocess: filter + z-score
        epoch = preprocess(epoch)

        #trim to match training data
        if epoch.shape[0] > min_samples:
            epoch = epoch[:min_samples, :]
        elif epoch.shape[0] < min_samples:
            pad = min_samples - epoch.shape[0]
            epoch = np.pad(epoch, ((0, pad), (0, 0)), mode='constant')

        #Reshape to (1, time, channels, 1) for Conv2D
        epoch = epoch.reshape(1, min_samples, len(eeg_channels), 1)

        probs = model.predict(epoch, verbose=0)
        label = label_map[int(np.argmax(probs))]
        print("Predicted label:", label)

        #Wait for the next epoch
        time.sleep(epoch_length)
        
except KeyboardInterrupt:
    print("Live classification stopped.")

finally:
    board.stop_stream()
    board.release_session()
