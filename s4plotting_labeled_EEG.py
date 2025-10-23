import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE = "filtered_eeg_action_data.csv"
TIME_COL = "BoardTime"
CH_COLS  = ["EEG_Ch1","EEG_Ch2","EEG_Ch3","EEG_Ch4"]
LABEL_COL= "Label"

filtered_eeg = pd.read_csv(FILE)
filtered_eeg = filtered_eeg.sort_values(["session_id", TIME_COL])

#continous label segments: returns [(label, start, stop), ...]
def label_segments(labels: np.ndarray, time: np.ndarray):
    change = labels != np.roll(labels, 1) #marks where labels are not the same, by rolling the label column by 1
    change[0] = True
    seg_id = np.cumsum(change) - 1
    for sid in np.unique(seg_id):
        m = (seg_id == sid)
        yield labels[m][0], time[m][0], time[m][-1] #finds where all the segments start and stop

#plotting a smaller number of points for speed, plot 10000 data points max with stride
def decimate_for_plot(df_sess: pd.DataFrame, max_pts: int = 10000):
    if len(df_sess) <= max_pts:
        return df_sess
    step = max(1, len(df_sess) // max_pts)
    return df_sess.iloc[::step].copy()

#plot one session
def plot_session(df_all: pd.DataFrame, session_id: int, title_suffix: str = "", max_plot_pts: int = 12000):
    g = df_all[df_all["session_id"] == session_id]
    if g.empty:
        print(f"No rows for session_id={session_id}")
        return
    g = decimate_for_plot(g, max_pts=max_plot_pts)

    time = g[TIME_COL].to_numpy()
    labels = g[LABEL_COL].astype(str).to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8),
                                   gridspec_kw={"height_ratios":[4,1]})

    # EEG traces
    for ch in CH_COLS:
        ax1.plot(time, g[ch].to_numpy(), linewidth=0.7, label=ch, rasterized=True)
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_title(f"Filtered EEG (0.5–20 Hz) — session {session_id}{title_suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # nice y-lims
    y = g[CH_COLS].to_numpy()
    ymin, ymax = float(np.min(y)), float(np.max(y))
    pad = 0.05 * (ymax - ymin + 1e-12)
    ax1.set_ylim(ymin - pad, ymax + pad)

    # label timeline
    for lab, t0, t1 in label_segments(labels, time):
        ax2.axvspan(t0, t1, alpha=0.28)
        ax2.text((t0+t1)/2, 0.5, lab, ha="center", va="center",
                 fontsize=9, transform=ax2.get_xaxis_transform())

    ax2.set_title("Label timeline")
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

#sessions i wanna inspect
session_ids = sorted(filtered_eeg["session_id"].unique())
for sid in session_ids[5:]: 
    plot_session(filtered_eeg, sid)