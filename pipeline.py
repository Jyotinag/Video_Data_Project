#Imports Here
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from tensorflow.keras import backend as K

#Meta_Data csv

meta_file = 'meta_data.csv'
meta_data = pd.read_csv(meta_file)
meta_data.head(5)

def feature_extract(input_video):
    cap = cv2.VideoCapture(input_video)

    # Define new frame size
    new_width, new_height = 224, 224

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        frames.append(resized_frame)

    cap.release()
    video_data = np.array(frames)
    i3d_model = InceptionV3(weights='imagenet', include_top=False)
    # Load and preprocess your video frames
    video_data_preprocessed = preprocess_input(video_data)
    # Extract features
    features = i3d_model.predict(video_data_preprocessed)
    return features

feature_list = np.empty((0, 5, 5, 2048))
label_list = []

count = 1

for index, row in meta_data.iterrows():
    video_path = row['video_clip']
    print('Video_Clips/clips'+video_path)
    feature_vector = feature_extract('Video_Clips/clips'+video_path)

    feature_list = np.concatenate((feature_list, feature_vector), axis=0)
    print(count)
    print(np.shape(feature_vector))
    count = count + 1

    label = row['cs_class']
    label_list.extend([label] * len(feature_vector))

labels_array = np.array(label_list)

reshaped_feature_data = feature_list.reshape((len(label_list), 5, 5 * 2048))

# Define the number of folds
k_folds = 10
target_data = label_list
target_data = np.array(target_data)
# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=False, random_state=42)
# Initialize lists to store results
test_losses = []
train_losses = []
val_losses = []
test_accuracies = []

# Iterate over the folds
for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(reshaped_feature_data, target_data)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    X_train, X_test = reshaped_feature_data[train_indices], reshaped_feature_data[test_indices]
    y_train, y_test = target_data[train_indices], target_data[test_indices]
    print(y_train)
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(5, 5 * 2048), return_sequences=True))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(LSTM(units=50))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Dense(units=4, activation='softmax'))
    # Train the model with dropout and early stopping
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
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