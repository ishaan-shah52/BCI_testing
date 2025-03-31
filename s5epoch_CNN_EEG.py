import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# -----------------------
# 1. Data Preprocessing & Epoching
# -----------------------

# Load the CSV file
df = pd.read_csv('filtered_eeg_action_data.csv')

# Rebase time so that the first time stamp is zero.
df['Time_Relative'] = df['Time (s)'] - df['Time (s)'].min()

# Define the desired epoch length (in seconds)
epoch_length = 3.0

# Create an epoch index by doing floor division on the relative time
df['Epoch_Index'] = (df['Time_Relative'] // epoch_length).astype(int)

# Group the data by epoch index
epochs = df.groupby('Epoch_Index')

# Collect epochs where the label remains consistent
consistent_epochs = []
for epoch_idx, group in epochs:
    unique_labels = group['Label'].unique()
    if len(unique_labels) == 1:
        consistent_epochs.append(group)
    else:
        print(f"Epoch {epoch_idx} discarded: multiple labels found {unique_labels}")

print(f"Collected {len(consistent_epochs)} consistent epochs.")

# -----------------------
# 2. Prepare Data for the CNN
# -----------------------

# Convert EEG data into a 2D format (time x channels)
X_list = []
y = []
for epoch in consistent_epochs:
    # Extract all available EEG channels as features (assuming they are labeled in the dataset)
    channel_cols = [col for col in df.columns if 'Filtered Channel' in col]
    epoch_data = epoch[channel_cols].values
    X_list.append(epoch_data)
    y.append(epoch['Label'].iloc[0])

# Determine the minimum number of samples across all epochs
min_samples = min(epoch.shape[0] for epoch in X_list)
print("Minimum samples per epoch:", min_samples)

# Truncate each epoch to have the same number of samples
X_fixed = [epoch[:min_samples] for epoch in X_list]

# Convert list of arrays into a single numpy array
X = np.array(X_fixed)  # Shape: (num_epochs, min_samples, num_channels)
y = np.array(y)

print("Shape of X (epochs):", X.shape)  # Expected: (num_epochs, time_samples, num_channels)
print("Unique labels:", np.unique(y))

# Map string labels to integer indices
label_map = {label: idx for idx, label in enumerate(np.unique(y))}
y_int = np.array([label_map[label] for label in y])
num_classes = len(label_map)
print("Label mapping:", label_map)

unique, counts = np.unique(y_int, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Convert labels to one-hot encoding
y_cat = to_categorical(y_int, num_classes=num_classes)

# Reshape X for CNN input (adding an extra channel dimension for Conv2D)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Shape: (num_epochs, time_samples, num_channels, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# -----------------------
# 3. Build the CNN (Following the Paper's Approach)
# -----------------------

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

