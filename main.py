
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

# Step 1: Load the MNIST dataset
print("Loading handwritten digit dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Step 2: Normalize image pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 3: Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Step 4: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 5: Train the model
print("Training the model...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the trained model
model.save("hcr_model.keras")

print("Model saved successfully!")

# Step 8: Predict a digit from the test dataset
index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(prediction[0])

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[index])

# Step 9: Display the test image
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()
''