import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers, callbacks

CSV = "filtered_eeg_action_data.csv"
TIME = "BoardTime"
CHANNELS = ["EEG_Ch1","EEG_Ch2","EEG_Ch3","EEG_Ch4"]
LABEL = "Label"
SESSION = "session_id"

FS = 50 #sampling frequency: downsampled to 50 Hz
WINDOW  = 2.0 # window length (seconds)
STRIDE_S = 0.5 #overlap between windows    
SPW = int(WINDOW * FS) #samples per window
SAMPLE_SLIDE = int(STRIDE_S * FS) #sample slide

filtered_eeg = pd.read_csv(CSV)

#label encoding: assigning labels to values
labels_sorted = sorted(filtered_eeg[LABEL].unique()) #arbitrary sort
label2id = {l:i for i,l in enumerate(labels_sorted)}
num_classes = len(labels_sorted)
print("Label map:", label2id)

#windowing by session
X, Y, S = [], [], [] # data is eeg windows, labels is the actions, keeping track of session
for sid, g in filtered_eeg.groupby(SESSION): #cant have sessions overlap
    g = g.reset_index(drop=True)
    n = len(g)
    lab_arr = g[LABEL].map(label2id).to_numpy() #string labels to int classes
    sig = g[CHANNELS].to_numpy(dtype=np.float32)  # shape (N, 4)

    start = 0
    while start + SPW <= n:
        seg = sig[start:start+SPW]          # (SPW, 4)
        labs = lab_arr[start:start+SPW]     # (SPW,)
        # Majority label in the window
        lab_id = np.bincount(labs, minlength=num_classes).argmax()

        # CNN expects (time, channels, 1) or (channels, time, 1).
        # use (time, channels, 1)
        X.append(seg[:, :].reshape(SPW, len(CHANNELS), 1))
        Y.append(lab_id)
        S.append(sid)   

        start += SAMPLE_SLIDE

X = np.stack(X)   # (num_windows, WIN, 4, 1)
Y = to_categorical(np.array(Y), num_classes=num_classes)
S = np.array(S)   # (num_windows,)
print("X shape:", X.shape, "Y shape:", Y.shape)


# Train/Val split (by session to prevent leakage)

# Split on session ids (not random windows)
unique_sess = np.unique(S)
train_sess, val_sess = train_test_split(unique_sess, test_size=0.25, random_state=42)
train_mask = np.isin(S, train_sess) #use a mask to find out which session in training or testing
val_mask   = np.isin(S, val_sess)

X_train, Y_train = X[train_mask], Y[train_mask]
X_val,   Y_val   = X[val_mask],   Y[val_mask]
print("Train windows:", len(X_train), "Val windows:", len(X_val))

# Minimal EEGNet-like model
# Input: (time, channels, 1)
#parameters: n_times is samples per window, channels, class labels
def build_eegnet_like(n_times, n_ch, n_classes,
                      F1=8, D=2, F2=16, drop1=0.25, drop2=0.25):
    inp = layers.Input(shape=(n_times, n_ch, 1))

    #start with temporal patterns like blink shape
    """
    Notes:
    -sampling rate: 50 hz --> 20 ms
    -doesnt mix with channels yet, just 16 samples: which should capture an action --> 16 * 20 ms = 320
    -F1 = 8: learns 8 temporal filters
    -no shrinking yet
    -batch normalization for each feature for consistent range, always after 2D conv
    --keeps activations in a good range, reduce internal covariate shift
    """
    x = layers.Conv2D(F1, kernel_size=(16, 1), padding='same', use_bias=False)(inp) 
    x = layers.BatchNormalization()(x)

    #spatial combine across channels
    """
    Notes:
    starting from (100, 4, 8)
    -each sample for all channels, 2 spatial filters per temporal feature
    --depthwise keeps parameters tiny instead of high
    -stabilization from batch normalization
    -elu works well with EEG for negative activations
    -pool time by 4x for nice even number to increase temporal receptive field for later
    -randomly drops features at end to prevent overfitting
    """
    x = layers.DepthwiseConv2D(kernel_size=(1, n_ch), depth_multiplier=D,
                               use_bias=False, depthwise_constraint=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D(pool_size=(4, 1))(x)
    x = layers.SpatialDropout2D(drop1)(x)

    #combines spatial mix with temporal aspect
    """
    Notes:
    -separableconv applies temporal filter to each input feature and mixes outputs
    --cuts parameters
    -stabilize with batch, pool by 4x again for recpetive field, shrink compute, dropout to help overfitting problem
    bascially looking at the temporal dimension for each spatial feature
    """
    x = layers.SeparableConv2D(F2, kernel_size=(16, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D(pool_size=(4, 1))(x)
    x = layers.SpatialDropout2D(drop2)(x)

    x = layers.Flatten()(x) #make into one final vector
    out = layers.Dense(n_classes, activation='softmax')(x) #probability for classification

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=1e-3) #good optimizer for adaptable gradient descent
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_eegnet_like(SPW, len(CHANNELS), num_classes)
model.summary()

#training
#call back for watching model durinng validation set, if accuracy isnt improving stop training to prevent overfitting
es = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')
hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# Evaluate
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"Val acc: {val_acc:.3f}")

model.save('EEG_CNN_model.h5')
print("Model saved as 'EEG_CNN_model.h5'")

