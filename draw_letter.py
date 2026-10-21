
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load the alphabet recognition model
model = load_model("hcr_letters_model.keras")

# Create the application window
root = tk.Tk()
root.title("Handwritten Alphabet Recognition")
root.geometry("500x600")
root.resizable(False, False)

# Drawing settings
CANVAS_SIZE = 400
BRUSH_SIZE = 18

# Create a white image for the model
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
draw = ImageDraw.Draw(image)

# Heading
title = tk.Label(
    root,
    text="Handwritten Alphabet Recognition",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

# Drawing canvas
canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    cursor="cross"
)
canvas.pack()

# Prediction result
result_label = tk.Label(
    root,
    text="Draw a capital letter (A-Z)",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=12)


def draw_letter(event):
    x, y = event.x, event.y
    r = BRUSH_SIZE // 2

    canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill="black",
        outline="black"
    )

    draw.ellipse(
        (x-r, y-r, x+r, y+r),
        fill=0
    )



def predict_letter():
    # Convert the drawing to a NumPy array
    img_array = np.array(image, dtype="uint8")

    # Convert black letters on white background
    # into white letters on black background
    img_array = 255 - img_array

    # Find the boundaries of the drawn letter
    rows, cols = np.where(img_array > 30)

    if len(rows) == 0:
        result_label.config(text="Please draw a letter first!")
        return

    # Crop around the letter
    cropped = img_array[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Resize while preserving the letter's proportions
    letter_img = Image.fromarray(cropped)
    letter_img.thumbnail((20, 20))

    # Place the letter in the center of a 28x28 image
    final_img = Image.new("L", (28, 28), 0)
    x = (28 - letter_img.width) // 2
    y = (28 - letter_img.height) // 2
    final_img.paste(letter_img, (x, y))

    # Normalize and prepare the image for prediction
    img_array = np.array(final_img, dtype="float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    letter = chr(predicted_index + ord("A"))

    result_label.config(
        text=f"Predicted Letter: {letter}\nConfidence: {confidence:.2f}%"
    )



def clear_canvas():
    global image, draw

    canvas.delete("all")
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(image)

    result_label.config(text="Draw a capital letter (A-Z)")


# Mouse drawing
canvas.bind("<B1-Motion>", draw_letter)
canvas.bind("<Button-1>", draw_letter)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="Predict Letter",
    command=predict_letter,
    font=("Arial", 12)
).pack(side=tk.LEFT, padx=10)

tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 12)
).pack(side=tk.LEFT, padx=10)

root.mainloop()
