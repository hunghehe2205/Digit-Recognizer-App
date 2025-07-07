import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
import io

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])


def preprocess_image(np_array):
    if np_array.dtype != np.uint8:
        np_array = (np_array * 255).astype(np.uint8)

    pil_image = Image.fromarray(np_array)

    return preprocess_pil_image(pil_image)


def preprocess_pil_image(pil_image):
    """
    Preprocess PIL Image for model inference
    """
    # Apply transforms
    tensor = transform(pil_image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor
