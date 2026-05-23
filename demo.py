import gradio as gr

def predict(image):
    return "Hi!"

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