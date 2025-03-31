import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------------------
# 1. Load and Epoch the Data
# ----------------------------

# Load the CSV file
data = pd.read_csv('filtered_eeg_action_data.csv')

# Rebase time so that the first timestamp is zero.
data['Time_Relative'] = data['Time (s)'] - data['Time (s)'].min()

# Define the epoch length (in seconds)
epoch_length = 2.0

# Create an epoch index by floor dividing the relative time by the epoch length
data['Epoch_Index'] = (data['Time_Relative'] // epoch_length).astype(int)

# Group the data by epoch
epochs = data.groupby('Epoch_Index')

# ----------------------------
# 2. Feature Extraction per Epoch
# ----------------------------
# We will only use epochs with a consistent label.
# For each epoch, compute the mean and standard deviation for 'Filtered Channel 2' and 'Filtered Channel 4'.

features = []
labels = []
for epoch_idx, group in epochs:
    unique_labels = group['Label'].unique()
    if len(unique_labels) == 1:
        # Compute features: mean and std of each channel.
        f_chan2_mean = group['Filtered Channel 2'].mean()
        f_chan2_std = group['Filtered Channel 2'].std()
        f_chan4_mean = group['Filtered Channel 4'].mean()
        f_chan4_std = group['Filtered Channel 4'].std()
        feature_vector = [f_chan2_mean, f_chan2_std, f_chan4_mean, f_chan4_std]
        features.append(feature_vector)
        labels.append(unique_labels[0])
    else:
        print(f"Epoch {epoch_idx} discarded: multiple labels found {unique_labels}")

X = np.array(features)
y = np.array(labels)

print("Feature matrix shape:", X.shape)
print("Unique labels:", np.unique(y))

# ----------------------------
# 3. Encode Labels and Split Data
# ----------------------------

# Encode string labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ----------------------------
# 4. Train the SVM Model
# ----------------------------
# Here we use a linear kernel SVM for simplicity.
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.2f}")

# ----------------------------
# 5. Save the Model for Live Classification
# ----------------------------

# Save both the SVM model and the label encoder for later use
model_data = {'model': svm, 'label_encoder': le}
joblib.dump(model_data, 'svm_model.joblib')
print("SVM model saved to 'svm_model.joblib'")
