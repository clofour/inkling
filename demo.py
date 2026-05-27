from shared import MODEL_DIR, IMAGE_SIZE, CATEGORIES
import random
import os.path as path
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models
import gradio as gr

MODEL_NAME = "sample"
MODEL_PATH = path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
CSS_FILE = "./demo.css"

model = models.load_model(MODEL_PATH)

def generate_blank_canvas():
    return Image.new("L", (280, 280), "white")

def generate_category():
    return random.choice(CATEGORIES)

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
    confidence = probabilities[predicted_index]

    if confidence >= 0.5:
        return f"{predicted_category} ({confidence:.2%})"
    else:
        return "No idea!"

def clear_all():
    return None, generate_blank_canvas(), None

with gr.Blocks() as demo:
    gr.Markdown("# Inkling")
    gr.Markdown("Draw the generated category!")

    with gr.Row():
        with gr.Column(scale=1):
            category_output = gr.Text(
                label="Category to draw",
                interactive=False
            )
            generate_category_button = gr.Button("Generate Category")
            generate_category_button.click(fn=generate_category, outputs=category_output)
            prediction_output = gr.Text(
                label="Prediction"
            )

        with gr.Column(scale=2):
            sketchpad_input = gr.Sketchpad(
                label="Canvas",
                value=generate_blank_canvas(),
                image_mode="L",
                brush=gr.Brush(
                    default_size=8,
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed"
                ),
                sources=[],
                buttons=[],
                transforms=[],
                layers=False
            )
            with gr.Row():
                submit_button = gr.Button("Submit")
                submit_button.click(fn=predict, inputs=sketchpad_input, outputs=prediction_output)
                clear_button = gr.Button("Clear")
                clear_button.click(fn=clear_all, outputs=[category_output, sketchpad_input, prediction_output])
        
demo.launch()