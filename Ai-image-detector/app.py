import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Real vs Fake Image Detector",
    page_icon="🔍",
    layout="centered"
)

# 2. DEFINE CONSTANTS
MODEL_PATH = 'best.pt'
CLASS_NAMES = ['Real', 'Fake']

# 3. MODEL LOADING FUNCTION (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    # Force CPU for deployment (safer for free cloud hosting tiers)
    device = torch.device('cpu')
    
    # Load the base architecture
    # Note: We don't need pretrained=True here since we are loading our own weights
    model = models.efficientnet_b0(weights=None)
    
    # Modify the classifier EXACTLY as you did in training
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 1)
    )
    
    # Load the state dict (weights)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PATH}' not found. Please place it in the same directory.")
        return None
        
    model.to(device)
    model.eval()
    return model

# 4. PREPROCESSING FUNCTION
def process_image(image):
    # Define the exact same transform used in your validation/inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# 5. UI LAYOUT
st.title("🕵️‍♂️ AI Image Detector")
st.write("Upload an image to check if it is **Real** or **AI Generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Load model
    model = load_model()
    
    if model:
        # Predict button
        if st.button('Analyze Image'):
            with st.spinner('Analyzing patterns...'):
                # Preprocess
                input_tensor = process_image(image)
                
                # Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    # Squeeze and apply sigmoid to get probability (0 to 1)
                    prob = torch.sigmoid(output).item()
                
                # Logic: In your training, > 0.5 was usually 'Fake' (Class 1)
                # Ensure this matches your specific labeling. 
                # Based on your code: real=0, fake=1.
                is_fake = prob >= 0.5
                confidence = prob if is_fake else (1 - prob)
                label = "AI Generated (Fake)" if is_fake else "Real Photography"
                
                # Display Result
                st.write("---")
                if is_fake:
                    st.error(f"## Result: {label}")
                else:
                    st.success(f"## Result: {label}")
                
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                st.progress(confidence)