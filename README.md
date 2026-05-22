# Intel Scene Image Classification

A PyTorch-based image classification project for recognizing natural scenes from the Intel Image Classification dataset. The goal is to build a reproducible computer vision project that trains a multiclass classifier, evaluates its behavior clearly, and prepares the model for deployment as a portfolio-ready application.

## Project Description

This project classifies scene images into one of six categories using a deep learning model trained on the Intel Image Classification dataset from Kaggle. It is designed as an end-to-end portfolio project, covering dataset preparation, preprocessing, training, evaluation, checkpoint saving, and later deployment.

## Dataset Overview

The Intel Image Classification dataset is a multiclass scene classification dataset from Kaggle. It contains natural scene images and is commonly described as having around 25,000 images of size 150x150 distributed across train, test, and prediction splits. The task is to predict which scene category an image belongs to.

## Classes

The dataset contains six scene categories:

- Buildings.
- Forest.
- Glacier.
- Mountain.
- Sea.
- Street.

## Image Sample
<img width="736" height="737" alt="image_samples" src="https://github.com/user-attachments/assets/43660945-df02-480e-aff0-ffdfcc9d81a0" />

## Image Sample
<img width="736" height="737" alt="image_samples" src="https://github.com/user-attachments/assets/43660945-df02-480e-aff0-ffdfcc9d81a0" />

## Project Structure

```text
.
├── README.md
├── .gitignore
├── environment.yml
├── requirements.txt
├── notebooks/
│   └── intel_classifier.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── model.py
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── models/
│   └── best_checkpoint_intel.pth
├── reports/
│   ├── confusion_matrix.png
│   ├── accuracy_vs_epochs.png
│   ├── loss_vs_epochs.png
│   ├── image_samples.png
│   ├── classification_report.png
│   └── sample_predictions.png
├── data/
│   └── intel_dataset/
└── assets/
```

This structure keeps notebooks, reusable code, model artifacts, reports, and raw data separate, which improves readability and reproducibility.

## Model Summary

The current notebook trains an image classification model in PyTorch using separate preprocessing pipelines for training and evaluation, including resizing, augmentation, tensor conversion, and normalization. The model is trained for six-way scene classification and should later be documented with the exact architecture name, optimizer, learning rate, image size, and checkpoint metadata in the final version of the project.

## Current Results

```text
- Best validation accuracy: 80.87%
- Best epoch: 53
- Architecture: VGG
- Input Image size: 128x128
- Notes: Common confusion between mountain and glacier
```

## Graphs
### Loss vs Epochs
<img width="1055" height="652" alt="loss_vs_epochs" src="https://github.com/user-attachments/assets/e32626e7-a015-4c23-a4ff-de6e442a8132" />

### Accuracy vs Epochs
<img width="1055" height="653" alt="accuracy_vs_epochs" src="https://github.com/user-attachments/assets/d05dad20-f5db-4879-977b-7c8d0128362f" />


## Classification Report
<img width="540" height="322" alt="classification_report" src="https://github.com/user-attachments/assets/fc1571be-bb6e-4ed4-9e1e-6ecc6733ffd2" />

## Confusion Matrix
<img width="702" height="616" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d223d998-fbe9-4b77-8cd3-771c434e784e" />

## Predictions on Unlabeled Data
<img width="751" height="768" alt="sample_predictions" src="https://github.com/user-attachments/assets/d5a8519e-e862-4c6c-a303-6ac733cc30fd" />


## How to Run

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies.
4. Download the Intel Image Classification dataset from Kaggle and place it under `data/intel_dataset/` with the expected folder structure.
5. Open the notebook or run the training script.

```bash
git clone https://github.com/andrewghaly27/intel-scene-image-classification
cd intel-scene-image-classification
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook notebooks/intel_classification.ipynb
```

## Deployment Architecture

This project uses a two-part deployment setup:

- **FastAPI backend** for model inference.
- **Streamlit frontend** for the user interface.

Workflow:
1. The user uploads an image in the Streamlit app.
2. The Streamlit app sends the image to the FastAPI backend.
3. The FastAPI backend preprocesses the image, loads the trained model checkpoint, and returns the top predictions.
4. The Streamlit app displays the predicted class and confidence scores.

## FastAPI Backend

The FastAPI backend is responsible for serving the trained model through an inference API.

**Local backend URL**
```text
http://127.0.0.1:8000
```

**Interactive API docs**
```text
http://127.0.0.1:8000/docs
```

## Streamlit Frontend

The Streamlit frontend provides the web interface for image upload and prediction display.

**Local frontend URL**
```text
http://localhost:8501
```

The frontend sends uploaded images to the FastAPI backend and displays the returned predictions.

## API Endpoints

### `GET /`
Health check endpoint.

**Example response**
```json
{
  "message": "Intel Scene Classification API is running"
}
```

### `POST /predict`
Accepts an uploaded image file and returns the top predicted scene classes with confidence scores.

**Example response**
```json
{
  "predictions": [
    {
      "class_name": "glacier",
      "confidence": 0.8421
    },
    {
      "class_name": "mountain",
      "confidence": 0.1134
    },
    {
      "class_name": "sea",
      "confidence": 0.0312
    }
  ]
}
```

## Run Locally

After cloning the repository and installing dependencies, run the backend and frontend in two separate terminals.

### 1. Start the FastAPI backend
```bash
uvicorn backend.main:app --reload
```

### 2. Start the Streamlit frontend
```bash
streamlit run frontend/app.py
```

### 3. Open the apps
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit UI: `http://localhost:8501`
