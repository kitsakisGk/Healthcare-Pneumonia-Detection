# AI-Powered Pneumonia Detection from Chest X-Rays

**Author:** Georgios Kitsakis
**Domain:** Healthcare AI, Medical Imaging, Deep Learning
**Status:** Research Project

## Overview

This project demonstrates an end-to-end deep learning pipeline for detecting pneumonia from chest X-ray images. The system combines state-of-the-art computer vision techniques with explainable AI to provide transparent, interpretable predictions suitable for medical applications.

## Motivation

Pneumonia is a leading cause of death worldwide, particularly in children under five. Early and accurate diagnosis is critical for effective treatment. This project aims to assist healthcare professionals by providing an AI-powered diagnostic tool that can:

- Automatically detect pneumonia from chest X-rays
- Highlight regions of interest using Grad-CAM visualization
- Provide confidence scores for predictions
- Offer a user-friendly interface for clinical use

## Key Features

- **Data Processing Pipeline**: Automated preprocessing, normalization, and augmentation
- **Baseline CNN**: Custom convolutional neural network architecture
- **Transfer Learning**: Fine-tuned ResNet50 for improved accuracy
- **Explainable AI**: Grad-CAM visualization showing model attention
- **Web Application**: Interactive Streamlit interface for real-time predictions
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, sensitivity, specificity

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

- **Total Images:** ~5,800 chest X-rays
- **Classes:** Normal, Pneumonia
- **Split:** Train/Test/Validation
- **Labels:** Validated by medical professionals

## Project Structure

```
healthcare-ai-pneumonia/
│
├── data/                   # Dataset (train/test/val folders)
├── notebooks/              # Jupyter notebooks for each phase
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── models/                 # Saved model weights
├── app/                    # Streamlit web application
│   ├── app.py
│   └── requirements.txt
├── reports/                # Results, plots, and analysis
│   ├── confusion_matrix.png
│   ├── gradcam_example.png
│   └── results_summary.md
└── README.md
```

## Tech Stack

- **Deep Learning:** PyTorch, TorchVision
- **Data Processing:** NumPy, Pandas, OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Explainability:** pytorch-grad-cam
- **Deployment:** Streamlit
- **Metrics:** scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-ai-pneumonia.git
cd healthcare-ai-pneumonia

# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

## Usage

### 1. Data Preprocessing
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

### 2. Model Training
```bash
jupyter notebook notebooks/02_training.ipynb
```

### 3. Evaluation & Grad-CAM
```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

### 4. Run Web Application
```bash
streamlit run app/app.py
```

## Results

| Metric | Baseline CNN | ResNet50 (Transfer Learning) |
|--------|--------------|------------------------------|
| Accuracy | ~87% | ~93% |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| F1-Score | TBD | TBD |

*Results will be updated after training completion*

## Model Explainability

This project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the X-ray the model focuses on when making predictions. This transparency is crucial for:

- Building trust with medical professionals
- Identifying potential model biases
- Validating clinical relevance of predictions

## Roadmap

- [x] Project setup and repository initialization
- [ ] Data exploration and preprocessing pipeline
- [ ] Baseline CNN model development
- [ ] Transfer learning with ResNet50
- [ ] Grad-CAM implementation
- [ ] Streamlit web application
- [ ] Results documentation and reporting

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
