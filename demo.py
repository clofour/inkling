import os.path as path
from PIL import Image
import numpy as np
from keras import models
import gradio as gr

MODEL_DIR = "./models"
MODEL_NAME = "20260523_1739"
MODEL_PATH = path.join(MODEL_DIR, f"{MODEL_NAME}.keras")

model = models.load_model(MODEL_PATH)

def process_image(image):
    pillow_image = Image.fromarray(image)
    pillow_image = pillow_image.convert("L")
    pillow_image = pillow_image.resize((28, 28), Image.Resampling.LANCZOS)

    np_image = np.array(pillow_image)
    np_image = 255 - np_image
    np_image = np_image / 255
    np_image = np_image.reshape(1, 28, 28, 1)

    return np_image

def predict(result):
    image = result["composite"]
    processed_image = process_image(image)
    prediction = model.predict(processed_image)
    return prediction

demo = gr.Interface(
    fn=predict,
    title="Drawing Classifier",
    description="Draw something!",
    inputs=[gr.ImageEditor(
        image_mode="L",
        sources=[],
        buttons=[],
        transforms=[],
        layers=False
    )],
    outputs=[gr.Text()]
)

demo.launch()