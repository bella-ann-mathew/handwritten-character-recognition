
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load the already trained model
model = load_model("hcr_model.keras")

# Create the drawing window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x500")
root.resizable(False, False)

canvas_size = 280

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    cursor="cross"
)
canvas.pack(pady=20)

# Image used to store the drawing
image = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(image)

last_x = None
last_y = None


def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y


def draw_digit(event):
    global last_x, last_y

    if last_x is not None and last_y is not None:
        canvas.create_line(
            last_x, last_y, event.x, event.y,
            fill="white",
            width=18,
            capstyle=tk.ROUND,
            smooth=True
        )

        draw.line(
            [last_x, last_y, event.x, event.y],
            fill=255,
            width=18
        )

    last_x = event.x
    last_y = event.y


def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None


def predict_digit():
    # Resize the drawing to the model's 28 x 28 input size
    resized = image.resize((28, 28))

    # Convert to a normalized array
    digit_array = np.array(resized).astype("float32") / 255.0
    digit_array = digit_array.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(digit_array, verbose=0)
    digit = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100

    result_label.config(
        text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%"
    )


def clear_canvas():
    global image, draw, last_x, last_y

    canvas.delete("all")
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    last_x = None
    last_y = None
    result_label.config(text="Draw a digit from 0 to 9")


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)

predict_button = tk.Button(
    root,
    text="Predict Digit",
    command=predict_digit,
    font=("Arial", 12)
)
predict_button.pack(pady=5)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 12)
)
clear_button.pack(pady=5)

result_label = tk.Label(
    root,
    text="Draw a digit from 0 to 9",
    font=("Arial", 14)
)
result_label.pack(pady=10)

root.mainloop()
