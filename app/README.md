# Streamlit Web Application

Interactive web interface for AI-powered pneumonia detection from chest X-rays.

## Features

- Upload chest X-ray images (JPEG/PNG)
- Real-time pneumonia detection using ResNet50
- Confidence scores for both classes (Normal/Pneumonia)
- Grad-CAM explainability heatmaps
- Download analysis reports

## Running the App

From the project root directory:

```bash
# Activate virtual environment (if using one)
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. Upload a chest X-ray image using the file uploader
2. Wait 2-3 seconds for analysis
3. View the prediction and confidence scores
4. Examine the Grad-CAM heatmap to see what the model focused on
5. Download the analysis report if needed

## Requirements

- Trained model file at `models/resnet50_pneumonia.pth`
- Python packages: `streamlit`, `torch`, `torchvision`, `pytorch-grad-cam`, `pillow`, `matplotlib`, `numpy`

## Disclaimer

This is a research/educational project. **NOT intended for clinical use.**
Always consult qualified healthcare professionals for medical diagnosis.
