import os
import logging
import tempfile
from pathlib import Path
from typing import NamedTuple
import gdown  # You need to install this package for downloading from Google Drive
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# Setting up Streamlit app configuration
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Logger setup
logger = logging.getLogger(__name__)

# Google Drive model URL (use your actual file ID here)
MODEL_GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1DdABmT5_axQpv6w51bfNwkq226A94D_m"

# Function to download model from Google Drive
def download_from_google_drive(url: str, destination: Path):
    """Download file from Google Drive using gdown."""
    if not destination.exists():
        logger.info(f"Downloading model from Google Drive to {destination}")
        gdown.download(url, str(destination), quiet=False)
    else:
        logger.info(f"Model already downloaded at {destination}")

# Function to cache and download the model to a temporary directory
@st.cache_resource
def load_model():
    # Create a temporary directory to store the model
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = Path(tmpdirname) / "YOLO_Pretrained_Model_RDD_FL.pt"
        
        # Download model if not already present
        download_from_google_drive(MODEL_GOOGLE_DRIVE_URL, model_path)
        
        # Load and return the model
        return YOLO(model_path)

# Load the model
net = load_model()

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# Detection class to store individual detection details
class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Streamlit UI
st.title("Road Damage Detection")
st.write("Please upload the image and start detecting")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

# Fixed confidence threshold set to 0.5 (no slider anymore)
score_threshold = 0.5
st.write("Confidence Threshold is fixed to 0.5 for detection.")

if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)

    # Perform inference
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)  # Using the fixed threshold
    
    # Process the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        downloadButton = st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
