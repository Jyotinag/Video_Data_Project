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

meta_file = 'meta_data.csv'
meta_data = pd.read_csv(meta_file)

meta_batch_size = 20
video_batch_size = 20

recent_model = ""
# Load and preprocess meta_data in batches
for meta_batch_num in range(0, len(meta_data), meta_batch_size):
    meta_batch = meta_data.iloc[meta_batch_num:meta_batch_num+meta_batch_size]

    # Initialize lists to store frames and labels for the current batch
    video_frames = []
    label_list = []

    # Iterate through meta_data batch
    for index, row in meta_batch.iterrows():
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
        label_list.extend([label] * len(frames))

    video_data = np.array(video_frames)
    video_data_preprocessed = preprocess_input(video_data)
    frames_array = np.array(video_data_preprocessed)
    labels_one_hot = to_categorical(label_list, num_classes=4)

    # Split the data into train and validation sets
    k_folds = 5
    target_data = label_list
    target_data = np.array(target_data)

    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=False, random_state=None)


    test_losses = []
    train_losses = []
    val_losses = []
    test_accuracies = []
    
    # Iterate over the folds
    for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(frames_array, target_data)):
        print(f'Fold {fold + 1}/{k_folds}')
        X_train, X_test = frames_array[train_indices], frames_array[test_indices]
        y_train, y_test = target_data[train_indices], target_data[test_indices]

        # Load or initialize the model
        if fold == 0 and meta_batch_num == 0:
            model = Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(4, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            # Load the model from the previous batch
            model = load_model(recent_model)

        # Train the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Save the model after each batch and fold
        save_model(model, f'model_batch_{meta_batch_num}_{fold}.h5')
        recent_model = f'model_batch_{meta_batch_num}_{fold}.h5'

        # Print the test loss for the current batch and fold
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Loss (Batch {meta_batch_num}-{meta_batch_num+meta_batch_size}, Fold {fold + 1}): {test_loss}')
        print(f'Test Accuracy (Batch {meta_batch_num}-{meta_batch_num+meta_batch_size}, Fold {fold + 1}): {test_accuracy}')
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # Store training and validation losses from the training history
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

    average_test_loss = np.mean(test_losses)
    print(f'Average Test Loss: {average_test_loss}')
    average_test_accuracy = np.mean(test_accuracies)
    print(f'Average Test Accuracy: {average_test_accuracy}')