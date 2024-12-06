import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model():
    # Load the test data
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    # Load the saved model
    model = tf.keras.models.load_model("model/model.h5")

    # Make predictions
    predictions = model.predict(x_test)
    predicted_classes = predictions.argmax(axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # Generate a classification report
    report = classification_report(y_test, predicted_classes, target_names=[str(i) for i in range(10)])
    print("Classification Report:\n", report)

if __name__ == "__main__":
    evaluate_model()
