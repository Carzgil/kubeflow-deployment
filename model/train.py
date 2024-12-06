import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_model():
    # Load preprocessed data
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_val = np.load("data/x_val.npy")
    y_val = np.load("data/y_val.npy")

    # Build a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

    # Save the model
    model.save("model/model.h5")
    print("Model training complete. Model saved as 'model/model.h5'.")

if __name__ == "__main__":
    train_model()
