import time
import threading
from collections import deque

import numpy as np
from pynput import keyboard
from scipy.signal import butter, sosfilt
from tensorflow.keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

#model
MODEL_PATH = "EEG_CNN_model.h5"

LABELS = ['both_blink', 'eyebrow_raise', 'left_blink', 'nothing', 'right_blink']

# Raw board rate and preprocessing
FS_RAW = 200
LOWCUT, HIGHCUT, ORDER = 0.5, 20.0, 4
FS = 50
DECIM = FS_RAW // FS

# Windowing (match training)
WINDOW_S = 2.0        # seconds
STRIDE_S = 0.5        # seconds
SPW = int(WINDOW_S * FS)   # 100
SAMPLE_SLIDE = int(STRIDE_S * FS)

# Set up BrainFlow parameters
params = BrainFlowInputParams()
params.serial_port = "COM5"      
params.mac_address = "D5:A4:BE:DD:BC:89" 

running = True
predict_every = 0

buf_50hz = deque(maxlen=SPW)  # rolling window at 50 Hz; each elem is shape (n_ch,)

model = None
board = None
eeg_channels = None
timestamps_channel = None

# Per-channel causal filter state (200 Hz, SOS)
sos = None
sos_z = None

#filters for live use
def bandpass_sos(lowcut, highcut, fs, order=4):
    #sos filter is important since numerically stable for live use and quick
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999999, highcut / nyq)
    return butter(order, [low, high], btype='band', output='sos')

#need this to stop
def on_release(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False  # stop listener
#summary: raw data -> filter -> downsample -> sliding window -> predicts
def record_and_predict():
    global predict_every, sos_z

    while running: #run until escape
        data = board.get_board_data()  # shape (channels x samples)
        n_new = data.shape[1] #how many samples
        if n_new == 0:
            time.sleep(0.01)
            continue

        # transpose to sample x channels
        x200 = data[eeg_channels, :].T.astype(np.float32)

        # Initialize SOS state on first run
        if sos_z is None:
            sos_z = [None] * x200.shape[1] 

        #filter each channel
        x200_f = np.empty_like(x200)
        for ch in range(x200.shape[1]):
            x200_f[:, ch], sos_z[ch] = sosfilt(sos, x200[:, ch], zi=sos_z[ch])

        # Downsample to 50 Hz by simple stride
        x50 = x200_f[::DECIM, :]  # (N/4, n_ch)

        # Feed samples into rolling buffer and predict every HOP
        for i in range(x50.shape[0]):
            buf_50hz.append(x50[i, :]) #holds a window
            predict_every += 1

            #2 seconds worth of data
            if len(buf_50hz) == SPW and predict_every >= SAMPLE_SLIDE:
                predict_every = 0

                # expected input shape (1, time, channels, 1)
                window = np.asarray(buf_50hz, dtype=np.float32)  # (SPW, n_ch)
                x_in = window.reshape(1, SPW, window.shape[1], 1)

                # If you normalized during training, apply the SAME here:
                # For example, per-window z-score (uncomment if used in training):
                # mu = x_in.mean(axis=(1, 2), keepdims=True)
                # sd = x_in.std(axis=(1, 2), keepdims=True) + 1e-6
                # x_in = (x_in - mu) / sd

                #pick class of highest probability
                p = model.predict(x_in, verbose=0)[0]  # softmax (n_classes,)
                pred_id = int(np.argmax(p))
                pred_label = LABELS[pred_id] if pred_id < len(LABELS) else str(pred_id)

                # Print compact live status
                print(f"[LIVE] {pred_label}  probs={np.round(p, 3)}")


def main():
    global model, board, eeg_channels, timestamps_channel, sos

    #instantiating later for safety incase hardware not set up yet
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    print("Preparing BrainFlow session...")
    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)
    timestamps_channel = BoardShim.get_timestamp_channel(BoardIds.GANGLION_BOARD.value)

    # Design causal SOS band-pass at 200 Hz
    sos = bandpass_sos(LOWCUT, HIGHCUT, FS_RAW, ORDER)

    board.prepare_session()
    board.start_stream(12000)

    # Start prediction thread, threading with main
    t = threading.Thread(target=record_and_predict, daemon=True)
    t.start()

    # ESC to stop
    listener = keyboard.Listener(on_release=on_release)
    listener.start()

    print("Live prediction running. Press ESC to stop.")
    try:
        while running and listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        print("Stopping...")
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
        running_local = False


if __name__ == "__main__":
    main()
