import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Save processed data to .npy files
    np.save("data/x_train.npy", x_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/x_val.npy", x_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/x_test.npy", x_test)
    np.save("data/y_test.npy", y_test)

    print("Preprocessing complete. Data saved.")

if __name__ == "__main__":
    preprocess_data()
