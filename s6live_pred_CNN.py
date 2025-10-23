import time
import threading
from collections import deque

import numpy as np
from pynput import keyboard
from scipy.signal import butter, sosfilt, sosfilt_zi
from tensorflow.keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


#model
MODEL_PATH = "EEG_CNN_model.h5"

LABELS = ['both_blink', 'left_blink', 'nothing', 'right_blink']

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

    while running:
        try:
            data = board.get_board_data()  # (n_rows x n_samples)
        except Exception:
            time.sleep(0.01)
            continue

        n_new = data.shape[1]
        if n_new == 0:
            time.sleep(0.01)
            continue

        # (n_samples, n_ch) float32
        x200 = data[eeg_channels, :].T.astype(np.float32)

        # init SOS zi on first block: one zi per channel
        if sos_z is None:
            zi0 = sosfilt_zi(sos)               # (n_sections, 2)
            sos_z = [zi0.copy() for _ in range(x200.shape[1])]

        # causal band-pass per channel at 200 Hz
        x200_f = np.empty_like(x200)
        for ch in range(x200.shape[1]):
            y, sos_z[ch] = sosfilt(sos, x200[:, ch], zi=sos_z[ch])
            x200_f[:, ch] = y

        # decimate → 50 Hz
        x50 = x200_f[::DECIM, :]  # (n_new/4, n_ch)

        # rolling buffer + stride
        for i in range(x50.shape[0]):
            buf_50hz.append(x50[i, :])
            predict_every += 1

            if len(buf_50hz) == SPW and predict_every >= SAMPLE_SLIDE:
                predict_every = 0

                # shape to (1, time, channels, 1)
                window = np.asarray(buf_50hz, dtype=np.float32)     # (SPW, n_ch)
                x_in = window.reshape(1, SPW, window.shape[1], 1)   # (1,100,C,1)

                # Per-window z-score (since we don't have mu/sd files)
                w_mu = x_in.mean(axis=(1, 2), keepdims=True)
                w_sd = x_in.std(axis=(1, 2), keepdims=True) + 1e-6
                x_in = (x_in - w_mu) / w_sd

                # Predict
                p = model.predict(x_in, verbose=0)[0]   # (n_classes,)
                pred_id = int(np.argmax(p))
                # Guard against mismatch between model classes and LABELS length
                if pred_id < len(LABELS):
                    label = LABELS[pred_id]
                else:
                    label = f"class_{pred_id}"
                print(f"[LIVE] {label:<13} probs={np.round(p, 3)}")

        time.sleep(0.005)

def main():
    global model, board, eeg_channels, timestamps_channel, sos, running

    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded. Classes:", model.output_shape[-1])

    print("Preparing BrainFlow session...")
    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)
    timestamps_channel = BoardShim.get_timestamp_channel(BoardIds.GANGLION_BOARD.value)

    # Design SOS band-pass at 200 Hz
    sos = bandpass_sos(LOWCUT, HIGHCUT, FS_RAW, ORDER)

    board.prepare_session()
    board.start_stream(12000)

    # Start prediction thread
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
        print("Stopping...")
        running = False
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass

if __name__ == "__main__":
    main()