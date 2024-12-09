import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam

def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

def train_model():
    # Load and normalize data
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_val = np.load("data/x_val.npy")
    y_val = np.load("data/y_val.npy")

    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0

    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,  #just for testing
        validation_data=(x_val, y_val),
        verbose=1
    )

    model.save("model/model.h5")
    print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
    return history

if __name__ == "__main__":
    train_model()
