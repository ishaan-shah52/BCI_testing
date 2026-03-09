# BCI EEG Codebase Overview

## What This Project Does

End-to-end pipeline for **EEG-based action classification** using an OpenBCI Ganglion board: record labeled EEG (blinks, eyebrow raise, nothing), clean/filter, train a small CNN (EEGNet-like), and run live predictions.

---

## Pipeline (Scripts in Order)

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `s0EEG_board_mac_finder.py` | BLE scan to find Ganglion MAC address |
| 1 | `s1predata_labeling_while_EEG.py` | Record EEG + keyboard labels (1–5) → `eeg_sessions/eeg_action_data_*.csv` |
| 2 | `s2combine_all_sessions.py` | Merge session CSVs, add `session_id` → `combined_eeg_data.csv` |
| 3 | `s3clean_python_EEG.py` | Bandpass 0.5–20 Hz, optional downsample to 50 Hz → `filtered_eeg_action_data.csv` |
| 4 | `s4plotting_labeled_EEG.py` | Plot filtered EEG + label timeline per session |
| 5 | `s5epoch_CNN_EEG.py` | Window data, train EEGNet-like CNN, save model + training curves |
| 6 | `s6live_pred_CNN.py` | Load model, stream from board, bandpass + downsample, sliding-window prediction |

**Auxiliary:**

- `labeling_data_external_soft.py` — Standalone keyboard label logger → `action_labels.csv` (no EEG; for syncing with external software).
- `plotting_EEG_channels.py` — Quick plot of `filtered_eeg_action_data.csv` (column indices may not match current CSV).
- `clean_openBCI_data.py` — Different pipeline: reads BrainFlow RAW TSV, selects columns → `neural_activity.csv`; not wired into s0–s6.

---

## Data Flow

```
s1: BoardTime, EEG_Ch1..4, Label (per sample)
    → eeg_sessions/eeg_action_data_1.csv, 2.csv, ...
s2: concat + session_id
    → combined_eeg_data.csv
s3: bandpass, downsample (200→50 Hz)
    → filtered_eeg_action_data.csv
s5: sliding windows (2 s, 0.5 s stride), majority label per window, train/val by session
    → EEG_CNN_model.h5
s6: same windowing + normalization at runtime
    → live predictions
```

---

## What’s Good

- **Session-aware splits** in s5 (train/val by `session_id`) reduce leakage.
- **Filtering** is correct: bandpass 0.5–20 Hz for eye/blink, zero-phase offline (s3), causal SOS in s6.
- **EEGNet-like model** (temporal → depthwise → separable) is appropriate for few channels.
- **Comments** in s5 explain filter lengths and pooling.
- **EarlyStopping** and validation in s5.

---

## Issues and How to Improve

### 1. **Critical: Label order mismatch (s5 vs s6)**

- **s5** builds classes from `sorted(filtered_eeg[LABEL].unique())` → alphabetical, e.g. `both_blink, eyebrow_raise, left_blink, nothing, right_blink`.
- **s6** uses a **hardcoded** list `['both_blink', 'left_blink', 'nothing', 'right_blink']` (wrong order, and **missing `eyebrow_raise`**).
- **Effect:** Live predictions map indices to wrong labels; 5-class models break in s6.

**Fix:** Save the label list (and optionally normalization stats) when training (e.g. next to `EEG_CNN_model.h5`). In s6, load that list and use it for `pred_id → label`. Implemented below.

---

### 2. **Fragile / magic strings**

- **s1:** Output path is hardcoded (`eeg_action_data_8.csv`); comment says “change these file numbers now” → easy to overwrite.
- **s3/s5:** Input file names and column names are literals in multiple scripts.

**Improvement:** Single config (e.g. `config.py` or `params.yaml`) for: paths, `CHANNELS`, `LABEL`, `SESSION`, `FS`, `WINDOW`, `STRIDE_S`, and label set. Scripts import from config.

---

### 3. **s5: No normalization stats saved for deployment**

- Training uses per-channel (and time) mean/std from the **training** set; s6 uses **per-window** z-score.
- **Effect:** Live distribution differs from training; possible accuracy drop.

**Improvement:** In s5, save `mu` and `sd` (e.g. in a small `.npz` or JSON). In s6, load and use them for normalization (with a fallback to per-window if file missing).

---

### 4. **s5: Ambiguous / low-confidence windows**

- Docstring suggests “take out non-majority windows like 60–70%” but it’s not implemented.
- **Improvement:** Add a `min_majority_ratio` (e.g. 0.8): only keep windows where one class has ≥ that fraction of samples. Optionally balance classes (e.g. oversample minority or cap majority).

---

### 5. **Window length vs. blinks**

- Docstring in s5: “blinks are shorter” — 2 s may be long for pure blink detection.
- **Improvement:** Make `WINDOW` (and optionally `STRIDE_S`) configurable; try e.g. 0.5–1 s for blink-only tasks and keep 2 s for “full context” if needed.

---

### 6. **plotting_EEG_channels.py**

- Uses column indices `[1,2]` and `[4,5]` and `applymap` (deprecated in pandas).
- **Improvement:** Use column names (`BoardTime`, `EEG_Ch1`–`EEG_Ch4`) and `map()` instead of `applymap()`.

---

### 7. **clean_openBCI_data.py**

- Different input/output and column layout; not part of s0–s6.
- **Improvement:** Either integrate it (e.g. as an alternate ingestion path) or move to an `experiments/` or `legacy/` folder and document.

---

### 8. **Dependencies and env**

- No `requirements.txt` or README.
- **Improvement:** Add `requirements.txt` (numpy, pandas, scipy, tensorflow, brainflow, pynput, bleak, matplotlib). Optional: README with setup and “run order” (s0 → s1 → … → s6).

---

### 9. **s1: Session naming**

- Output filename is manually changed each run; risk of overwriting.
- **Improvement:** Auto-increment or use timestamp in filename (e.g. `eeg_action_data_20250308_143022.csv`).

---

### 10. **Testing and reproducibility**

- No tests or fixed seeds in some scripts.
- **Improvement:** Set `random_state` / `np.random.seed` and `tf.random.set_seed` in s5; add a small test that loads model + label list and runs one batch.

---

## Suggested Run Order

1. Run **s0** once to get MAC; put it in s1/s6 (and config if you add it).
2. Run **s1** per recording session; adjust output filename or use auto-naming.
3. Run **s2** after you have at least two session files.
4. Run **s3** to produce `filtered_eeg_action_data.csv`.
5. Optionally run **s4** to inspect sessions.
6. Run **s5** to train and save `EEG_CNN_model.h5` (and label list + optional norm stats).
7. Run **s6** for live prediction (load label list from s5 output).

---

## Summary Table of Improvements

| Area | Priority | Action |
|------|----------|--------|
| Label order s5↔s6 | Critical | Save/load label list with model; use in s6 |
| Normalization in s6 | High | Save train mu/sd in s5; load in s6 |
| Config / constants | Medium | Single config for paths, channels, FS, window, labels |
| Majority-only windows | Medium | Add min_majority_ratio in s5 |
| requirements.txt / README | Medium | Add deps and short setup/run instructions |
| plotting_EEG_channels | Low | Use column names, fix deprecated API |
| s1 output naming | Low | Auto-increment or timestamp |
| clean_openBCI_data | Low | Document or relocate |
