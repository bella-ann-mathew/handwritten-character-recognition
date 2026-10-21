
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Step 1: Load EMNIST Letters
print("Loading EMNIST Letters dataset...")

train_data, test_data = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    data_dir="tfds_data"
)

# Step 2: Convert dataset to NumPy arrays
def prepare_data(dataset):
    images = []
    labels = []

    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int64")

    return images, labels


x_train, y_train = prepare_data(train_data)
x_test, y_test = prepare_data(test_data)

print("Dataset loaded!")
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Step 3: Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# EMNIST images may be transposed relative to normal image orientation
x_train = np.transpose(x_train, (0, 2, 1, 3))
x_test = np.transpose(x_test, (0, 2, 1, 3))

# EMNIST Letters labels are 1-26; convert to 0-25
y_train = y_train - 1
y_test = y_test - 1

# Step 4: Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(26, activation="softmax")
])

# Step 5: Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 6: Train model
print("Training alphabet model...")

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# Step 7: Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"Letter test accuracy: {accuracy * 100:.2f}%")

# Step 8: Save alphabet model separately
model.save("hcr_letters_model.keras")

print("Alphabet model saved successfully!")
