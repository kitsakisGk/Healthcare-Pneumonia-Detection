# AI-Powered Pneumonia Detection from Chest X-Rays

**Author:** Georgios Kitsakis
**Domain:** Healthcare AI, Medical Imaging, Deep Learning
**Status:** ✅ Complete - Production Ready
**GitHub:** [Healthcare-Pneumonia-Detection](https://github.com/kitsakisGk/Healthcare-Pneumonia-Detection)

## Overview

This project demonstrates an end-to-end deep learning pipeline for detecting pneumonia from chest X-ray images. The system combines state-of-the-art computer vision techniques with explainable AI (Grad-CAM) to provide transparent, interpretable predictions suitable for medical applications. Features a complete web application for real-time inference.

## Motivation

Pneumonia is a leading cause of death worldwide, particularly in children under five. Early and accurate diagnosis is critical for effective treatment. This project aims to assist healthcare professionals by providing an AI-powered diagnostic tool that can:

- Automatically detect pneumonia from chest X-rays
- Highlight regions of interest using Grad-CAM visualization
- Provide confidence scores for predictions
- Offer a user-friendly interface for clinical use

## Key Features

✅ **Complete End-to-End Pipeline**
- Data exploration and preprocessing with augmentation
- Transfer learning with ResNet50 (ImageNet pre-trained)
- 89.58% test accuracy achieved
- Training time: 226 minutes on CPU

✅ **Explainable AI**
- Grad-CAM heatmap visualizations
- Shows model attention regions on X-rays
- Validates clinical relevance of predictions

✅ **Interactive Web Application**
- Upload and analyze X-rays in real-time
- Confidence scores for both classes
- Downloadable analysis reports
- User-friendly interface with medical disclaimers

✅ **Professional Documentation**
- Comprehensive Jupyter notebooks for each phase
- Detailed visualizations and metrics
- Clean git history with meaningful commits

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

- **Total Images:** 5,856 chest X-rays
- **Training Set:** 5,216 images (NORMAL: 1,341 | PNEUMONIA: 3,875)
- **Validation Set:** 16 images (NORMAL: 8 | PNEUMONIA: 8)
- **Test Set:** 624 images (NORMAL: 234 | PNEUMONIA: 390)
- **Classes:** Binary classification (Normal vs Pneumonia)
- **Image Format:** JPEG, grayscale
- **Resolution:** Variable (224x224 after preprocessing)

## Project Structure

```
AI-Powered-Disease-Detection/
│
├── data/                      # Dataset (organized by train/test/val)
│   ├── train/                 # 5,216 training images
│   ├── test/                  # 624 test images
│   └── val/                   # 16 validation images
│
├── notebooks/                 # Complete pipeline notebooks
│   ├── 01_preprocessing.ipynb    # Data exploration & preprocessing
│   ├── 02_training.ipynb         # ResNet50 training (89.58% accuracy)
│   └── 03_gradcam.ipynb          # Explainability visualizations
│
├── models/                    # Trained model weights
│   └── resnet50_pneumonia.pth    # Final trained model (220 MB)
│
├── app/                       # Web application
│   ├── streamlit_app.py          # Interactive web interface
│   └── README.md                 # App usage instructions
│
├── reports/                   # Visualizations and results
│   ├── training_curves.png       # Loss/accuracy over epochs
│   ├── confusion_matrix.png      # Classification metrics
│   ├── class_distribution.png    # Dataset statistics
│   ├── augmentation_examples.png # Data augmentation samples
│   ├── gradcam_normal_*.png      # Grad-CAM for normal cases
│   └── gradcam_pneumonia_*.png   # Grad-CAM for pneumonia cases
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Tech Stack

- **Deep Learning:** PyTorch 2.8.0, TorchVision 0.20.0
- **Model Architecture:** ResNet50 (transfer learning with ImageNet weights)
- **Data Processing:** NumPy, Pandas, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Explainability:** pytorch-grad-cam 1.6.0 (Grad-CAM heatmaps)
- **Web Application:** Streamlit 1.30.0
- **Development:** Jupyter Notebook, Python 3.9+

## Installation

```bash
# Clone the repository
git clone https://github.com/kitsakisGk/Healthcare-Pneumonia-Detection.git
cd Healthcare-Pneumonia-Detection

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The trained model file (`models/resnet50_pneumonia.pth`) is required to run the web application. Download the dataset from Kaggle and place it in the `data/` folder to retrain from scratch.

## Usage

### Running the Web Application

The quickest way to see the project in action:

```bash
# From project root directory
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`. Upload a chest X-ray image to get instant predictions with Grad-CAM visualizations!

### Reproducing the Pipeline

Follow the notebooks in order to reproduce the complete pipeline:

**Phase 1: Data Exploration & Preprocessing**
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```
- Analyzes dataset statistics and class distribution
- Creates augmentation examples
- Saves visualizations to `reports/`

**Phase 2: Model Training**
```bash
jupyter notebook notebooks/02_training.ipynb
```
- Trains ResNet50 with transfer learning
- Takes ~226 minutes on CPU (faster with GPU)
- Achieves 89.58% test accuracy
- Saves model to `models/resnet50_pneumonia.pth`

**Phase 3: Model Explainability**
```bash
jupyter notebook notebooks/03_gradcam.ipynb
```
- Generates Grad-CAM heatmaps
- Creates visualization comparisons
- Saves results to `reports/`

## Results

### Final Model Performance

**Model:** ResNet50 with Transfer Learning (ImageNet pre-trained)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.58%** |
| **Training Time** | 226 minutes (CPU) |
| **Epochs** | 5 (with early stopping patience) |
| **Best Validation Accuracy** | 87.50% (Epoch 3) |
| **Final Test Loss** | 0.2871 |

### Training Details

- **Architecture:** ResNet50 with custom classifier head
- **Transfer Learning:** Froze layers 1-3, fine-tuned layer 4 + FC layers
- **Optimizer:** Adam (lr=0.0001)
- **Loss Function:** CrossEntropyLoss with class weights (1:3 ratio)
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Data Augmentation:** Random rotation, horizontal flip, color jitter
- **Regularization:** Dropout (0.5, 0.3), early stopping

### Model Insights

The confusion matrix and Grad-CAM visualizations show that the model:
- Successfully focuses on lung regions (clinically relevant)
- Identifies infiltrates and consolidations in pneumonia cases
- Maintains high sensitivity for pneumonia detection
- Provides transparent, explainable predictions via Grad-CAM

## Model Explainability

This project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the X-ray the model focuses on when making predictions. This transparency is crucial for:

- Building trust with medical professionals
- Identifying potential model biases
- Validating clinical relevance of predictions

## Project Phases - Complete

- [x] **Phase 0:** Project setup and repository initialization
- [x] **Phase 1:** Data exploration and preprocessing pipeline
- [x] **Phase 2:** Transfer learning with ResNet50 (89.58% accuracy)
- [x] **Phase 3:** Grad-CAM explainability implementation
- [x] **Phase 4:** Streamlit web application deployment
- [x] **Phase 5:** Complete documentation and results reporting

All phases completed successfully with clean git history!

## Ethical Considerations

This project is intended for **research and educational purposes only**. It should not be used as a replacement for professional medical diagnosis. Key considerations:

- Model predictions should always be validated by qualified healthcare professionals
- Patient data privacy and confidentiality must be maintained
- Potential biases in training data should be acknowledged
- Regulatory compliance (e.g., HIPAA, GDPR) is essential for clinical deployment

## Future Enhancements

- Multi-class classification (COVID-19, tuberculosis, etc.)
- Integration with DICOM medical imaging standards
- Model deployment on cloud platforms (AWS, Azure, GCP)
- Mobile application for point-of-care use
- Integration with hospital information systems

## References

- [CheXNet: Radiologist-Level Pneumonia Detection](https://arxiv.org/abs/1711.05225)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)

## License

MIT License - See LICENSE file for details

## Contact

**Georgios Kitsakis**
For questions or collaboration opportunities, please reach out via GitHub Issues.

---

*This project demonstrates proficiency in AI/ML engineering, data science, medical imaging analysis, and software deployment - skills valuable for healthcare technology roles in Switzerland and beyond.*
