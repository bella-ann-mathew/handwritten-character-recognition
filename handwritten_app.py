
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load both existing models
digit_model = load_model("hcr_model.keras")
letter_model = load_model("hcr_letters_model.keras")

# Window setup
root = tk.Tk()
root.title("Handwritten Character Recognition")
root.geometry("520x650")
root.resizable(False, False)

CANVAS_SIZE = 400
BRUSH_SIZE = 18

# Drawing image
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
draw = ImageDraw.Draw(image)

# Heading
tk.Label(
    root,
    text="Handwritten Character Recognition",
    font=("Arial", 17, "bold")
).pack(pady=10)

# Mode selection
mode = tk.StringVar(value="digit")

tk.Label(
    root,
    text="Choose what you want to recognize:",
    font=("Arial", 11)
).pack()

mode_frame = tk.Frame(root)
mode_frame.pack(pady=5)

tk.Radiobutton(
    mode_frame,
    text="Digits (0-9)",
    variable=mode,
    value="digit"
).pack(side=tk.LEFT, padx=10)

tk.Radiobutton(
    mode_frame,
    text="Letters (A-Z)",
    variable=mode,
    value="letter"
).pack(side=tk.LEFT, padx=10)

# Drawing canvas
canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    cursor="cross"
)
canvas.pack(pady=8)

result_label = tk.Label(
    root,
    text="Draw a digit or capital letter",
    font=("Arial", 15, "bold")
)
result_label.pack(pady=8)


def draw_character(event):
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


def prepare_image():
    # Convert black strokes on white background
    # into white strokes on black background
    pixels = 255 - np.array(image, dtype="uint8")

    rows, cols = np.where(pixels > 30)

    if len(rows) == 0:
        return None

    # Crop to the drawn character
    cropped = pixels[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    character = Image.fromarray(cropped)
    character.thumbnail((20, 20))

    # Center in a 28x28 image
    final_image = Image.new("L", (28, 28), 0)
    x = (28 - character.width) // 2
    y = (28 - character.height) // 2
    final_image.paste(character, (x, y))

    array = np.array(final_image, dtype="float32") / 255.0
    return array.reshape(1, 28, 28, 1)



def predict_character():
    processed = prepare_image()

    if processed is None:
        result_label.config(text="Please draw something first!")
        return

    if mode.get() == "digit":
        # Keep digit preprocessing unchanged
        prediction = digit_model.predict(processed, verbose=0)
        index = int(np.argmax(prediction))
        character = str(index)

    else:
        # Match the orientation used when training EMNIST
        letter_input = processed

        prediction = letter_model.predict(letter_input, verbose=0)
        index = int(np.argmax(prediction))
        character = chr(index + ord("A"))

    confidence = float(np.max(prediction)) * 100

    result_label.config(
        text=f"Prediction: {character}  |  Confidence: {confidence:.2f}%"
    )


def clear_canvas():
    global image, draw

    canvas.delete("all")
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(image)

    result_label.config(text="Draw a digit or capital letter")


canvas.bind("<B1-Motion>", draw_character)
canvas.bind("<Button-1>", draw_character)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="Predict",
    command=predict_character,
    font=("Arial", 12),
    width=12
).pack(side=tk.LEFT, padx=10)

tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 12),
    width=12
).pack(side=tk.LEFT, padx=10)
print("Alphabet model output shape:", letter_model.output_shape)
print("Alphabet model input shape:", letter_model.input_shape)
root.mainloop()
