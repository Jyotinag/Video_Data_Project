# Imports Here
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load InceptionV3 without top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained layers

def save_model(model, filename):
    model.save(filename)

# Function to load the saved model
def load_model(filename):
    return tf.keras.models.load_model(filename)
# Add Custom Classification Head

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Adjust the number of units for your 4 classes
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess meta_data
meta_file = 'meta_data.csv'
meta_data = pd.read_csv(meta_file)

# Initialize lists to store frames and labels
video_frames = []
label_list = []

# Iterate through meta_data
for index, row in meta_data.iterrows():
    video_path = 'Video_Clips/clips' + row['video_clip']
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (224, 224))
        video_frames.append(resized_frame)
        frames.append(resized_frame)

    cap.release()
    

    # Store labels
    label = row['cs_class']
    #print(label)
    label_list.extend([label] * len(frames))

video_data = np.array(video_frames)

    # Preprocess and store video frames
video_data_preprocessed = preprocess_input(video_data)
#video_frames.append(video_data_preprocessed)
# Preprocess data and labels
frames_array = np.array(video_data_preprocessed)
labels_one_hot = to_categorical(label_list, num_classes=4)

# Split the data into train and validation sets
#print(len(label_list))
k_folds = 10
target_data = label_list
target_data = np.array(target_data)
# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=False, random_state=None)
# Initialize lists to store results
test_losses = []
train_losses = []
val_losses = []
test_accuracies = []

# Iterate over the folds
for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(frames_array, target_data)):
    print(f'Fold {fold + 1}/{k_folds}')
    X_train, X_test = frames_array[train_indices], frames_array[test_indices]
    y_train, y_test = target_data[train_indices], target_data[test_indices]

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Append results to lists
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    test_losses.append(test_loss)
    
    # Store training and validation losses from the training history
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

# Calculate and print the average test loss and accuracy across folds
average_test_loss = np.mean(test_losses)
print(f'Average Test Loss: {average_test_loss}')

# Plot training, validation, and test losses
plt.figure(figsize=(12, 6))
for i in range(k_folds):
    plt.plot(train_losses[i], label=f'Training Fold {i+1}')
    plt.plot(val_losses[i], label=f'Validation Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Across Folds')
plt.legend()
plt.show()
plt.savefig('met.png')
