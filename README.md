# Digit-Recognizer-App
A complete machine learning inference system featuring a PyTorch implementation of the classic LeNet architecture from scratch with Model Training,  FastAPI backend and Gradio frontend for interactive image classification.

## 🎯 Motivation

This project demonstrates a production-ready deployment of a deep learning model, showcasing:

- **Classic Architecture Implementation**: LeNet-5, one of the pioneering convolutional neural networks
- **Full-Stack ML Pipeline**: From model inference to web-based user interface
- **Interactive Classification**: Support for both image upload and drawing canvas
- **Modern Web Technologies**: FastAPI for robust API endpoints and Gradio for intuitive UI

Perfect for learning deployment patterns, experimenting with CNN architectures, or as a foundation for more complex computer vision applications.
## 🏗️ Project Structure
```
├── app
│   ├── api
│   │   └── main.py
│   ├── front_end
│   │   └── gradio_app.py
│   ├── models
│   │   └── model.py
│   └── utils
│       └── image_processing.py
├── requirements.txt
├── run.py
└── weights
    ├── LeNet_MNIST.ipynb
    └── lenet_model.pt

```
### Key Components

- **`app/models/model.py`**: Contains the LeNet-5 CNN architecture with customizable output classes
- **`app/api/main.py`**: FastAPI server providing `/predict` and `/predict_test` endpoints
- **`app/front_end/gradio_app.py`**: Interactive web interface with upload and drawing capabilities
- **`app/utils/image_processing.py`**: Image preprocessing pipeline (grayscale, resize, normalize)
- **`run.py`**: Multi-process launcher for both backend and frontend services
- **`weights/LeNet_MNIST`**: MNIST training notebook
  ### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pytorch-lenet-classifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare model weights**:
   - Place your trained `lenet_model.pt` file in the `weights/` directory
   - The model should be trained for 10 classes (default configuration)

### Running the Application

#### Option 1: Full Application (Recommended)
```bash
python run.py
```

This launches both services:
- **FastAPI Backend**: `http://localhost:8000`
- **Gradio Frontend**: `http://localhost:7860`

#### Option 2: Individual Services

**Backend only**:
```bash
python -m app.api.main
# or
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

**Frontend only** (requires backend running):
```bash
python -m app.front_end.gradio_app
```

### Usage

1. **Web Interface**: Open `http://localhost:7860` in your browser
2. **Upload Tab**: Upload images for classification
3. **Draw Tab**: Use the drawing canvas to create images
4. **API Endpoints**: 
   - `GET /health` - Health check
   - `POST /predict` - JSON array input (used by Gradio)
   - `POST /predict_test` - File upload endpoint

## 🛠️ Technical Details

### Model Architecture
- **LeNet-5 CNN**: 2 convolutional layers + 3 fully connected layers
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (configurable)
- **Preprocessing**: Grayscale conversion, resize, normalization

### API Specification

**POST /predict**
```json
{
  "data": [[[pixel_values]]]  // 3D array: height x width x channels
}
```

**Response**
```json
{
  "predicted_class": 5,
  "confidence": 0.95,
  "probabilities": [0.01, 0.02, ...]
}
```

### Frontend Features
- **Dual Input Methods**: File upload or drawing canvas
- **Real-time Prediction**: Automatic classification on input change
- **Probability Visualization**: Class confidence scores
- **Responsive Design**: Clean, intuitive interface
