import gradio as gr
import requests
import numpy as np
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fastapi_url = "http://localhost:8000"


import gradio as gr
import requests
import numpy as np
import logging
from PIL import Image


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fastapi_url = "http://localhost:8000"


def predict(image):
    """
    Send image to FastAPI for prediction
    """
    if image is None:
        return "Please provide an image", {}
    
    try:
        # Handle different input types from Gradio components
        if isinstance(image, dict):
            # This is from Sketchpad - it returns a dict with composite image
            if 'composite' in image and image['composite'] is not None:
                # Convert PIL Image to numpy array
                pil_image = image['composite']
                image_array = np.array(pil_image)
            else:
                return "Please draw something on the canvas", {}
        elif hasattr(image, 'shape'):
            # This is already a numpy array
            image_array = image
        else:
            # Convert to numpy array
            image_array = np.array(image)
        
        # Ensure the image is in the right format
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            image_array = image_array[:, :, :3]
        
        # Convert to list for JSON serialization
        image_list = image_array.tolist()
        
        # Send request to FastAPI
        response = requests.post(
            f"{fastapi_url}/predict",
            json={"data": image_list},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()

            # Format result
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            # Create probability dictionary for gradio
            prob_dict = {f"Class {i}": prob for i,
                         prob in enumerate(probabilities)}

            prediction_text = f"Predicted Class: {predicted_class}"

            return prediction_text, prob_dict
        else:
            return f"Error: {response.status_code} - {response.text}", {}

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return f"Connection error: {str(e)}", {}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Prediction error: {str(e)}", {}


def create_interface():
    """
    Create and return Gradio interface
    """
    with gr.Blocks(title="PyTorch LeNet Classifier") as interface:
        gr.Markdown("# PyTorch LeNet Image Classifier")
        gr.Markdown(
            "Upload an image or draw in the canvas below to classify it using our LeNet model.")

        with gr.Tab("Upload Image"):
            with gr.Row():
                with gr.Column():
                    image_upload = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Upload Image"
                    )
                    upload_btn = gr.Button(
                        "Classify Uploaded Image", variant="primary")

                with gr.Column():
                    upload_output = gr.Textbox(label="Prediction Result")
                    upload_plot = gr.Label(label="Class Probabilities")

        with gr.Tab("Draw Image"):
            with gr.Row():
                with gr.Column():
                    canvas = gr.Sketchpad(
                        sources=[],
                        type="numpy",
                        label="Draw Your Image",
                        canvas_size=(280, 280)
                    )
                    draw_btn = gr.Button("Classify Drawing", variant="primary")

                with gr.Column():
                    draw_output = gr.Textbox(label="Prediction Result")
                    draw_plot = gr.Label(label="Class Probabilities")

        # Event handlers
        upload_btn.click(
            fn=predict,
            inputs=[image_upload],
            outputs=[upload_output, upload_plot]
        )

        draw_btn.click(
            fn=predict,
            inputs=[canvas],
            outputs=[draw_output, draw_plot]
        )

        # Auto-predict on image change
        image_upload.change(
            fn=predict,
            inputs=[image_upload],
            outputs=[upload_output, upload_plot]
        )

        canvas.change(
            fn=predict,
            inputs=[canvas],
            outputs=[draw_output, draw_plot]
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )