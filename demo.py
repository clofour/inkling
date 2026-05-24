from shared import MODEL_DIR, IMAGE_SIZE, CATEGORIES
import random
import os.path as path
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models
import gradio as gr

MODEL_NAME = "20260523_2000"
MODEL_PATH = path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
CSS_FILE = "./demo.css"

model = models.load_model(MODEL_PATH)

def process_image(image):
    pillow_image = Image.fromarray(image)
    pillow_image = pillow_image.convert("L")
    pillow_image = pillow_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    np_image = np.array(pillow_image)
    np_image = 255 - np_image
    np_image = np_image / 255
    np_image = np_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return np_image

def predict(result):
    image = result["composite"]
    processed_image = process_image(image)

    logits = model.predict(processed_image)
    probabilities = tf.nn.softmax(logits[0]).numpy()
    predicted_index = np.argmax(probabilities)
    predicted_category = CATEGORIES[predicted_index]

    return predicted_category

def generate_category():
    return random.choice(CATEGORIES)

with gr.Blocks(css_paths=CSS_FILE) as demo:
    gr.Markdown("# Drawing Classifier")
    gr.Markdown("Draw the generated category!")

    category_output = gr.Text(
        label="Category to draw",
        interactive=False
    )
    generate_category_button = gr.Button("Generate Category")
    generate_category_button.click(fn=generate_category, outputs=category_output)

    sketchpad_input = gr.Sketchpad(
        image_mode="L",
        sources=[],
        buttons=[],
        transforms=[],
        layers=False
    )
    submit_button = gr.Button("Submit")
    prediction_output = gr.Text(
        label="Prediction"
    )
    submit_button.click(fn=predict, inputs=sketchpad_input, outputs=prediction_output)
    clear_button = gr.ClearButton(components=[category_output, sketchpad_input, prediction_output])

demo.launch()