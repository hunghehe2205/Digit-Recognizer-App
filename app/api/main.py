import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import logging
from typing import List

from app.models.model import LeNetClassifier
from app.utils.image_processing import preprocess_image, preprocess_pil_image


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PyTorch LeNet Inference API")

# Enable CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list
    
# Add input model for the predict endpoint
class ImageInput(BaseModel):
    data: List[List[List[float]]]  # 3D array for the image data
    
    
@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = os.path.join("weights", "lenet_model.pt")
        
        # Initialize the model
        model = LeNetClassifier(num_classes=10)
        
        if os.path.exists(model_path):
            #map_location : load state dict to the correct device
            state_dict = torch.load(model_path, map_location=device, weights_only=True) # dictionary of model parameters
            
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            raise HTTPException(status_code=500, detail="Model file not found")
        
        # Move model to the appropriate device compatiable with state_dict
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")
    
@app.get("/")
async def root():
    return {"message": "PyTorch LeNet Inference API is running!"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }
    
@app.post("/predict_test", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class of uploaded image from FastAPI endpoint.
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        inputs = preprocess_pil_image(image) #return tensor with shape (1, 1, 28, 28)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1) # convert logits to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item() # get the index of the class with highest probability
            confidence = probabilities[0][predicted_class].item() # get the probability of the predicted class
            return PredictionResponse(
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities[0].cpu().numpy().tolist()
            )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    

@app.post("/predict", response_model=PredictionResponse)
async def predict_array(input: ImageInput):
    """ 
    Predict class of input numpy array from Gradio frontend.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        np_array = np.array(input.data, dtype=np.float32)
        
        input_tensor = preprocess_image(np_array)  # Returns tensor with shape (1, 1, 28, 28)
        input_tensor = input_tensor.to(device)
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 

