# CodeAlpha Task 3: Handwritten Character Recognition

## Project Overview

This project recognizes handwritten digits (0–9) and English alphabet letters (A–Z) using deep learning. It provides a graphical interface where users can draw a character and receive a prediction from the trained model.

## Features

- Recognizes handwritten digits from 0 to 9.
- Recognizes handwritten English letters from A to Z.
- Provides a combined graphical interface for digit and letter recognition.
- Allows users to draw characters using a canvas.
- Includes Predict and Clear functionality.
- Uses separately trained models for digits and letters.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- TensorFlow Datasets
- Tkinter
- MNIST dataset for digit recognition
- EMNIST Letters dataset for alphabet recognition

## Project Files

- `main.py` — trains the digit recognition model.
- `draw_digit.py` — provides the digit recognition interface.
- `train_letters.py` — trains the alphabet recognition model.
- `draw_letter.py` — provides the alphabet recognition interface.
- `handwritten_app.py` — combines digit and alphabet recognition.
- `test_letters.py` — tests the alphabet model.
- `hcr_model.keras` — trained digit recognition model.
- `hcr_letters_model.keras` — trained alphabet recognition model.
- `requirements.txt` — Python dependency list.

## Setup Instructions

1. Install Python.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment if needed.
4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Ensure the trained model files are present in the project directory.

## Run the Application

Run the combined recognition application:

```bash
python handwritten_app.py
```

Select **Digits** to recognize a number or **Letters** to recognize an English alphabet character. Draw a character on the canvas and click **Predict**. Use **Clear** before drawing another character.

## Testing

The combined application was tested with handwritten digit and alphabet inputs. Both recognition modes produced correct predictions in the tested examples.

## Project Purpose

The project demonstrates how deep learning can be used for handwritten character recognition through image preprocessing, model training, prediction, and a graphical user interface.