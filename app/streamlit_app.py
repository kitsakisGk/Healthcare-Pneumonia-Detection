"""
AI-Powered Pneumonia Detection Web Application

Author: Georgios Kitsakis
Date: 2025-10-29

Description:
Interactive web app for pneumonia detection from chest X-rays using ResNet50
with Grad-CAM explainability visualizations.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Load model
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    # Load trained weights
    model.load_state_dict(torch.load('../models/resnet50_pneumonia.pth', map_location=device))
    model = model.to(device)
    model.eval()

    return model

# Load Grad-CAM
@st.cache_resource
def load_gradcam(_model):
    """Initialize Grad-CAM with the model"""
    target_layers = [_model.layer4[-1]]
    cam = GradCAM(model=_model, target_layers=target_layers)
    return cam

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert to grayscale first
    if image.mode != 'L':
        image = image.convert('L')

    # For visualization
    img_np = np.array(image.resize((224, 224)))
    img_rgb = np.stack([img_np, img_np, img_np], axis=-1) / 255.0

    # For model
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    return input_tensor, img_rgb

def predict_with_gradcam(model, cam, image):
    """Make prediction and generate Grad-CAM"""
    input_tensor, img_rgb = preprocess_image(image)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

        # Get both class probabilities
        normal_prob = probabilities[0][0].item() * 100
        pneumonia_prob = probabilities[0][1].item() * 100

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay heatmap
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    return predicted_class, confidence, normal_prob, pneumonia_prob, visualization, img_rgb

# Main app
def main():
    st.markdown('<div class="main-header">🫁 AI-Powered Pneumonia Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Model with Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **ResNet50** deep learning model trained on chest X-ray images
        to detect pneumonia.

        **Features:**
        - Binary classification (Normal vs Pneumonia)
        - Transfer learning with ImageNet pre-training
        - Grad-CAM explainability visualizations
        - 89.58% test accuracy

        **Dataset:**
        - Training: 5,216 images
        - Validation: 16 images
        - Test: 624 images
        """)

        st.markdown("---")

        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a chest X-ray image (JPEG/PNG)
        2. Wait for analysis (~2-3 seconds)
        3. View prediction and confidence
        4. Examine Grad-CAM heatmap

        **Grad-CAM Colors:**
        - 🔴 Red/Hot: High attention
        - 🔵 Blue/Cold: Low attention
        """)

        st.markdown("---")

        st.header("⚠️ Disclaimer")
        st.warning("""
        This is a **research project** for educational purposes.

        **NOT for clinical use.** Always consult qualified medical
        professionals for diagnosis and treatment.
        """)

        st.markdown("---")
        st.markdown("**Author:** Georgios Kitsakis  ")
        st.markdown("**Model:** ResNet50 + Transfer Learning  ")
        st.markdown("**Framework:** PyTorch 2.8.0")

    # Main content
    try:
        model = load_model()
        cam = load_gradcam(model)
        st.success(f"✓ Model loaded successfully! Running on: **{device.type.upper()}**")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model file exists at: `../models/resnet50_pneumonia.pth`")
        return

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image in JPEG or PNG format"
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Create columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original X-Ray")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔬 Analysis")
            with st.spinner("Analyzing image... Please wait..."):
                try:
                    # Make prediction
                    pred_class, confidence, normal_prob, pneumonia_prob, gradcam_img, original_img = predict_with_gradcam(model, cam, image)

                    # Display prediction
                    class_names = ['NORMAL', 'PNEUMONIA']
                    prediction = class_names[pred_class]

                    if pred_class == 0:  # NORMAL
                        st.markdown(
                            f'<div class="prediction-box normal">✓ {prediction}</div>',
                            unsafe_allow_html=True
                        )
                    else:  # PNEUMONIA
                        st.markdown(
                            f'<div class="prediction-box pneumonia">⚠ {prediction}</div>',
                            unsafe_allow_html=True
                        )

                    # Confidence metrics
                    st.markdown("### 📊 Confidence Scores")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Normal", f"{normal_prob:.2f}%")
                    with col_b:
                        st.metric("Pneumonia", f"{pneumonia_prob:.2f}%")

                    # Progress bars
                    st.progress(normal_prob / 100, text=f"Normal: {normal_prob:.1f}%")
                    st.progress(pneumonia_prob / 100, text=f"Pneumonia: {pneumonia_prob:.1f}%")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    return

        # Grad-CAM visualization
        st.markdown("---")
        st.markdown("## 🎯 Explainability: Grad-CAM Heatmap")

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **What is Grad-CAM?**
        Gradient-weighted Class Activation Mapping shows which regions of the X-ray
        the AI model focuses on to make its prediction. Red/hot areas indicate high
        attention, while blue/cold areas show low attention.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📷 Original")
            st.image(original_img, use_container_width=True)

        with col4:
            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(gradcam_img, use_container_width=True)

        # Interpretation guide
        st.markdown("---")
        st.markdown("### 💡 Interpretation Guide")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""
            **✓ Good Signs:**
            - Heatmap focuses on lung fields
            - Attention on infiltrates/consolidations (if pneumonia)
            - More diffuse attention (if normal)
            """)

        with col6:
            st.markdown("""
            **⚠ Warning Signs:**
            - Focus on image edges or corners
            - Attention on non-anatomical features
            - May indicate spurious correlations
            """)

        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")

        # Create downloadable figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('Original X-Ray', fontsize=13, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(gradcam_img)
        ax2.set_title('Grad-CAM Heatmap', fontsize=13, fontweight='bold')
        ax2.axis('off')

        fig.suptitle(
            f'Prediction: {prediction} ({confidence:.1f}% confidence)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.download_button(
            label="📥 Download Analysis Report (PNG)",
            data=buf,
            file_name=f"pneumonia_analysis_{prediction.lower()}.png",
            mime="image/png"
        )

        plt.close()

    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")

        st.markdown("---")
        st.markdown("### 🖼️ Sample Images")
        st.markdown("""
        You can test the application with sample images from the test dataset:
        - Normal cases: `data/test/NORMAL/`
        - Pneumonia cases: `data/test/PNEUMONIA/`
        """)

if __name__ == "__main__":
    main()
