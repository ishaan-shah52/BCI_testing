import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

filtered_eeg = pd.read_csv('filtered_eeg_action_data.csv')

#start time at 0 for simplicity of analysis
# filtered_eeg['Time_Relative'] = filtered_eeg['Time'] - filtered_eeg['Time'].iloc[0]
filtered_eeg['Time_Relative'] = filtered_eeg['Time'] - filtered_eeg['Time'].min()

#Two seconds for each sample
epoch_length = 2.0

#Making buckets with floor division
filtered_eeg['Epoch_Index'] = (filtered_eeg['Time_Relative'] // epoch_length).astype(int)
epochs = filtered_eeg.groupby('Epoch_Index')

min_len_required = 12         # 3 rows = kernel height of first Conv2D

consistent_epochs = []
for epoch_idx, group in epochs:
    one_label = len(group['Label'].unique()) == 1 #cant mix labels, might change this, but for now just discard if mixed
    long_enough = len(group) >= min_len_required #has to have a certain amount of rows for kernel   
    if one_label and long_enough:
        consistent_epochs.append(group)
    #debugging
    else:
        print(f"Epoch {epoch_idx} discarded "
              f"(rows={len(group)}, labels={group['Label'].unique()})")

print(f"Collected {len(consistent_epochs)} consistent epochs.")

# training and labeled data
X_list = []
y = []
for epoch in consistent_epochs:
    # Extract all available EEG channels as features
    channel_cols = [col for col in filtered_eeg.columns if 'Filtered Channel' in col]
    epoch_data = epoch[channel_cols].values
    X_list.append(epoch_data)
    y.append(epoch['Label'].iloc[0])

# maintain consistency with minimum number of samples per epoch
min_samples = min(epoch.shape[0] for epoch in X_list) #first num is rows
print("Minimum samples per epoch:", min_samples)

X_fixed = [epoch[:min_samples] for epoch in X_list] #truncate

X = np.array(X_fixed)  # Shape: (num_epochs, min_samples, num_channels)
y = np.array(y)

print("Shape of X (epochs):", X.shape)  # Expected: (num_epochs, time_samples, num_channels)
print("Unique labels:", np.unique(y))

print("check here")

# Map string labels to integer indices for CNN
label_map = {label: idx for idx, label in enumerate(np.unique(y))}
y_int = np.array([label_map[label] for label in y])
num_classes = len(label_map)
print("Label mapping:", label_map)

unique, counts = np.unique(y_int, return_counts=True) #gives a count for appearance of all
print("Class distribution:", dict(zip(unique, counts))) #need to see if data points for each label distributed equally, dict makes readable

#one-hot encoding
y_cat = to_categorical(y_int, num_classes=num_classes)

"""
Using Conv2D for spatial and temporal instead of Conv1D just for time per channel 
"""

# Conv2D expects (batch_size, height, width, channels) or (amount of epochs, time samples, channels, 1)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Shape: (num_epochs, time_samples, num_channels, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]  # (time_samples, num_channels, 1)

model = Sequential()

# First Layer (Spatial feature extraction)
model.add(Conv2D(filters=8, kernel_size=(3, X.shape[2]), activation='relu', input_shape=input_shape, padding='valid'))
model.add(BatchNormalization())

# Second Layer (Temporal feature extraction)
model.add(SeparableConv2D(filters=16, kernel_size=(5, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))

# Third Layer (Deeper feature extraction)
model.add(Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Train with a Learning Rate Scheduler
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)])

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

model.save('EEG_CNN_model.h5')
print("Model saved as 'EEG_CNN_model.h5'")

