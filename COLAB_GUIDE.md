# Google Colab Setup Guide

**Project:** AI-Powered Pneumonia Detection
**Author:** Georgios Kitsakis

## Why Use Google Colab?

- ✅ **FREE GPU Access** (Tesla T4/K80)
- ✅ **10x-20x Faster Training** than CPU
- ✅ **No Local Setup Required**
- ✅ **Save Models to Google Drive**

---

## Quick Start (3 Steps)

### Step 1: Upload Dataset to Google Drive

1. Download dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. Extract the zip file
3. Upload to Google Drive:
   ```
   MyDrive/
   └── pneumonia_data/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── test/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── val/
           ├── NORMAL/
           └── PNEUMONIA/
   ```

### Step 2: Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `03_transfer_learning.ipynb`
4. **OR** directly from GitHub:
   - Click **File → Open Notebook → GitHub**
   - Paste: `https://github.com/kitsakisGk/Healthcare-Pneumonia-Detection`
   - Select `03_transfer_learning.ipynb`

### Step 3: Enable GPU

1. Click **Runtime → Change Runtime Type**
2. Select **Hardware Accelerator: GPU**
3. Click **Save**

---

## Running the Notebook

### 1. Mount Google Drive (First Cell)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link, authorize, paste the code.

### 2. Update Data Path (Second Cell)

Change the `BASE_DIR` to match your Google Drive path:

```python
BASE_DIR = '/content/drive/MyDrive/pneumonia_data'
```

### 3. Run All Cells

Click **Runtime → Run All**

The notebook will automatically:
- Detect GPU
- Load pre-trained ResNet50
- Train the model (10-15 minutes on GPU!)
- Save best model to your Drive
- Generate evaluation metrics
- Create visualizations

---

## Expected Training Time

| Device | Time |
|--------|------|
| **Colab GPU (T4)** | **10-15 minutes** ⚡ |
| Colab CPU | 2-3 hours |
| Local CPU | 2-4 hours |

---

## After Training

### Results will be saved to:
```
MyDrive/pneumonia_models/
└── resnet50_best.pth

MyDrive/pneumonia_reports/
├── training_curves_resnet50.png
├── confusion_matrix_resnet50.png
└── results_resnet50.md
```

### Download Results:
1. Open Files panel (left sidebar)
2. Navigate to `/content/drive/MyDrive/pneumonia_models/`
3. Right-click → Download

---

## Tips & Tricks

### 1. Check GPU is Working
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

### 2. Monitor Training
- Watch the progress bars in real-time
- Training accuracy should reach 95%+ by epoch 5

### 3. Save Your Work
- Models automatically save to Google Drive
- Colab sessions timeout after 12 hours
- Download important results to local machine

### 4. Re-run if Disconnected
If Colab disconnects:
1. Reconnect to runtime
2. Re-run from the training cell
3. Model will resume from checkpoint

---

## Alternative: Upload Dataset Directly to Colab

If you don't want to use Google Drive:

```python
# Upload dataset zip to Colab
from google.colab import files
uploaded = files.upload()  # Select chest-xray-pneumonia.zip

# Extract
!unzip chest-xray-pneumonia.zip -d /content/data/

# Update BASE_DIR
BASE_DIR = '/content/data/chest_xray'
```

**Note:** Colab storage is temporary - lost when session ends!

---

## Troubleshooting

### Problem: "Runtime Disconnected"
**Solution:** Colab has usage limits. Wait 30 minutes and try again.

### Problem: "Out of Memory"
**Solution:** Reduce `BATCH_SIZE` from 32 to 16 in config cell.

### Problem: "Dataset Not Found"
**Solution:** Double-check your Google Drive path in `BASE_DIR`.

### Problem: "Slow Training on Colab"
**Solution:** Make sure GPU is enabled (Runtime → Change Runtime Type → GPU).

---

## Results You Should Expect

With ResNet50 on GPU:

- **Test Accuracy:** 92-95%
- **Sensitivity:** 95-97%
- **Specificity:** 85-90%
- **Training Time:** 10-15 minutes

This is production-level performance! 🎉

---

## Next Steps After Training

1. ✅ Download trained model
2. ✅ Download visualizations
3. ✅ Review results summary
4. ✅ Move to Phase 4: Grad-CAM (explainability)
5. ✅ Build Streamlit web app

---

**Happy Training!** 🚀

For questions or issues, check the [GitHub repo](https://github.com/kitsakisGk/Healthcare-Pneumonia-Detection).
