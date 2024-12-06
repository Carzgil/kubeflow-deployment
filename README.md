# CIFAR-10 ML Pipeline with Kubeflow

This project demonstrates the creation and deployment of a scalable machine learning pipeline using **Kubeflow Pipelines (KFP)**. The pipeline processes the CIFAR-10 dataset, trains a Convolutional Neural Network (CNN) model, and evaluates its performance. The pipeline is designed to be modular, portable, and scalable.

## Features
- **Preprocessing**: Normalizes and splits the CIFAR-10 dataset into training, validation, and test sets.
- **Training**: Trains a CNN model on the preprocessed data.
- **Evaluation**: Evaluates the trained model's accuracy and generates a classification report.
- **Containerized Components**: Each stage of the pipeline is containerized for reproducibility and portability.
- **Kubeflow Integration**: The pipeline is deployed and managed using Kubeflow Pipelines.

---

## Project Structure

```plaintext
kubeflow-pipeline-project/
├── data/                       # Data-related scripts and files
│   ├── preprocessing.py        # Preprocessing script
├── model/                      # Model-related scripts and files
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── model.h5                # Trained model file
├── pipeline/                   # Pipeline definition and components
│   ├── pipeline.py             # Kubeflow pipeline definition
│   ├── components/             # Pipeline components
│       ├── preprocess_component.py
│       ├── train_component.py
│       ├── evaluate_component.py
├── docker/                     # Dockerfiles for each component
│   ├── Dockerfile.preprocess
│   ├── Dockerfile.train
│   ├── Dockerfile.evaluate
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignore unnecessary files
```

## Prerequisites

1. **Environment Setup**:
   - Python 3.9+
   - Docker
   - Kubernetes cluster with Kubeflow installed
   - `kubectl` and `kfp` Python SDK installed

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt

---

## How to Run Locally

### 1. Preprocess the Data
Run the preprocessing script to generate the `.npy` files:
```bash
python data/preprocessing.py
```
### 2. Train the Model
Run the training script to train the model:
```bash
python model/train.py
```
### 3. Evaluate the Model
Run the evaluation script to evaluate the model:
```bash
python model/evaluate.py
```

## How to build Docker Images
```bash
docker build -t preprocess-image docker/Dockerfile.preprocess
docker build -t train-image docker/Dockerfile.train
docker build -t evaluate-image docker/Dockerfile.evaluate
```

## How to Run Kubeflow Pipeline

### 1. Compile the Pipeline
```bash
python pipeline/pipeline.py
```
### 2. Upload to Kubeflow Pipelines
- Open Kubeflow Pipelines UI
- Click on "Upload Pipeline"
- Select the `cifar10_pipeline.yaml` file
- Click "Run" to execute the pipeline
