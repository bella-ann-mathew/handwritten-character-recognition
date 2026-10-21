
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

print("Loading test data...")

test_data = tfds.load(
    "emnist/letters",
    split="test",
    as_supervised=True,
    data_dir="tfds_data"
)

model = load_model("hcr_letters_model.keras")

# Test a few images directly from the dataset
for image, label in tfds.as_numpy(test_data.take(10)):
    image = image.astype("float32") / 255.0

    # Match the orientation used during training
    image = np.transpose(image, (1, 0, 2))

    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image, verbose=0)
    predicted_index = int(np.argmax(prediction))

    # TFDS EMNIST Letters labels are normally 1-26
    actual_letter = chr(int(label) + ord("A") - 1)
    predicted_letter = chr(predicted_index + ord("A"))

    print(
        f"Actual: {actual_letter} | "
        f"Predicted: {predicted_letter} | "
        f"Confidence: {np.max(prediction) * 100:.2f}%"
    )
