import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# -------------------------------
# 1. Load the Saved LDA Model
# -------------------------------
# The model_data contains both the trained LDA model and the label encoder.
model_data = joblib.load('lda_model.joblib')
lda = model_data['model']
label_encoder = model_data['label_encoder']

# -------------------------------
# 2. Set Up the Data Acquisition (BrainFlow)
# -------------------------------
params = BrainFlowInputParams()
# Adjust these parameters as needed for your device
params.serial_port = "COM5"            # or the port your device uses
params.mac_address = "D5:A4:BE:DD:BC:89" # if needed for your device
board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
board.prepare_session()
board.start_stream()

# Define epoch parameters
epoch_length = 2.0  # seconds
fs = board.get_sampling_rate(BoardIds.GANGLION_BOARD.value)
samples_per_epoch = int(epoch_length * fs)

print("Starting live LDA classification. Press Ctrl+C to stop.")

try:
    while True:
        # --------------------------------
        # 3. Acquire a Single Epoch of Data
        # --------------------------------
        # Get the latest data for one epoch.
        # board.get_current_board_data returns an array with shape (num_channels, num_samples)
        data = board.get_current_board_data(samples_per_epoch)
        
        # -------------------------------
        # 4. Extract Features from the Epoch
        # -------------------------------
        # Here, we assume that columns 4 and 5 contain 'Filtered Channel 2' and 'Filtered Channel 4'.
        # Adjust the channel indices as needed.
        selected_channels = [4, 5]
        # Extract data from the selected channels.
        epoch_data = data[selected_channels, :]  # Shape: (2, samples_per_epoch)
        
        # Compute features: mean and standard deviation for each channel.
        feature_vector = []
        for channel_data in epoch_data:
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            feature_vector.extend([mean_val, std_val])
        
        # Convert to a 2D array with shape (1, 4) for prediction.
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # --------------------------------
        # 5. Make a Prediction with LDA
        # --------------------------------
        prediction = lda.predict(feature_vector)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print("Predicted label:", predicted_label)
        
        # Wait for the next epoch
        time.sleep(epoch_length)

except KeyboardInterrupt:
    print("Live classification stopped.")

finally:
    board.stop_stream()
    board.release_session()
