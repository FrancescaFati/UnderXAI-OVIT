# UnderXAI-OVIT: OC Segmentation Analysis & PCS Classification Pipeline

![Model Architecture](model_architecture.png)

This repository implements a full pipeline for ovarian cancer (OC) image analysis and tumor resectability classification, developed as part of the UnderXAI-OVIT project. It integrates segmentation-based feature extraction, image pre-processing, model training, and evaluation to support clinical decision-making.

Key components include:

- **Lesion feature extraction** from NIfTI segmentation files, including metrics such as volume, surface area, compactness, and fractal dimension.
- **3D image pre-processing** to normalize and structure CT volumes for downstream analysis.
- **Deep learning model training and evaluation** for predicting resectability from combined imaging and clinical features.
- **Bootstrapped performance analysis** to ensure robust and statistically sound metric estimation.


## Getting Started

Follow these instructions to set up the project on your local machine for development and testing.

### Prerequisites

- Python 3.7 or higher
- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda recommended)

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/your_username/UnderXAI-OVIT.git
    cd UnderXAI-OVIT
    ```

2. **Create and activate a Conda environment**

    ```bash
    conda create -n underxai python=3.9  # Use Python >= 3.7
    conda activate underxai
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```
---

## 0. Image Preprocessing
This module handles preprocessing of 3D CT images, including automatic OC segmentation, format conversion, and visualization.

### 0.1 OV-segmentation Inference
Run lesion segmentation on a list of NIfTI images using pretrained OV-Seg models.

**Usage:**
```bash
python ov_seg.py your_image_paths.txt --models pod_om 
```
- `your_image_paths.txt`: Path to the text file containing paths to NIfTI images.
- `--models`: Name(s) of models used during inference. Options are: `pod_om`, `abdominal_lesions`, `lymph_nodes`. Can combine multiple.
- `--fast`: (Optional) Speeds up inference speed by disabling dynamic z-spacing, model ensembling and test-time augmentations.

### 0.2 NIfTI to PTH Conversion
Convert and preprocess NIfTI images into PyTorch tensors (.pth) using MONAI transforms, including intensity windowing and resizing.

**Usage:**
```bash
python pth.py your_nifti_image_paths.txt --output_dir ./processed_data
```
- `your_nifti_image_paths.txt`: Path to the text file containing NIfTI image paths(output from OV-segmentation.
- `--output_dir`:  Root directory for the output files.

### 0.3 Visualization of NIfTI or PTH Files
Use the provided Jupyter notebook for visual inspection of CT images and segmentations.
- `Notebook`: visualization.ipynb
Supports both raw and segmented CT volumes in `.nii` or `.pth` format
Interactive viewer with slice navigation and overlay controls

---

## 1. Lesion Feature Extraction

This module computes a set of quantitative features from a NIfTI segmentation file and exports the results to a CSV file.

Extracted features include:
- **Volume**
- **Surface area**
- **Compactness**
- **Fractal dimension**

**Usage:**
```bash
python lesion_features_from_nifti.py --nifti /path/to/your_segmentation.nii.gz --output /path/to/output.csv
```
- `--nifti`: Path to the local NIfTI file containing the segmentation mask.
- `--output`: Destination path for the output CSV file containing the computed features.

---

## 2. Main Pipeline: Training & Evaluation
Train and evaluate the tumor resectability classifier using both clinical and imaging features.

**Prepare a config file** (YAML format, see `config.yaml` for reference):
```yaml
# --- General Training Settings ---
accumulation_steps: 16
attention_resc: 2
batch_size: 2
clinical_encoding_dim: 4
clinical_features: True
clinical_input_dim: 4
dropout: 0.25
early_stopping_patience: 70
epochs: 1
feature_extractor: "google-32"
learning_rate: 0.0001
loss_function: "WBCE"
num_heads: 4
optimizer: "adamW"
scheduler_gamma: 0.5
scheduler_patience: 60
scheduler_step: 5
scheduler_type: "step"
weight_decay: 1.0e-6

# --- Data Paths ---
train_paths: "/home/ffati/UnderXAI-OVIT/ds.json"
val_paths: "/home/ffati/UnderXAI-OVIT/ds.json"
test_paths: "/home/ffati/UnderXAI-OVIT/ds.json"
```
Each dataset file (e.g., ds.json) should contain preprocessed .pth tensors and clinical features.

**Run training and evaluation:**
```bash
python main.py --config config.yaml
```
This will:

- Train the model using the specified configuration.
- Save the best model checkpoint (based on validation loss) in the `checkpoints/` directory.
- Save the final trained model in the `models/` directory.
- Evaluate the model on the test set and store the results in the `test/` directory.

---
## 3. Model Testing & Bootstrapping
Evaluate a trained model on the test set using bootstrapping to obtain robust estimates of performance metrics.

Before running the script, update the following variables in `test.py`:
- `TEST_DATA_PATH`: Path to the test dataset JSON file.
- `CONFIG_YAML_PATH`: Path to the YAML configuration used during training.
- `CHECKPOINT_PATH`: Path to the saved model checkpoint.

---

**Run the Evaluatin:**
```bash
python test.py
```
- This will run bootstrapped evaluation and save metrics and predictions as CSV files.

---

## 4. Configuration Files
- `config.yaml` and `config_sweep.yaml` provide example settings for training and hyperparameter sweeps.

---

## 5. Directory Structure
```
UnderXAI-OVIT/
├── main.py                # Main training/evaluation pipeline
├── test.py                # Bootstrapped model evaluation
├── lesion_features_from_nifti.py  # Lesion feature extraction
├── ov_seg.py              # OV-segmentation inference script
├── pth.py                 # NIfTI to PTH conversion script
├── model.py, engine.py, pcs_dataset.py, metrics.py, utilis.py  # Core modules for model definition, training loop, dataset handling, evaluation metrics, and utility functions.
├── config.yaml, config_sweep.yaml # Example configurations for training and hyperparameter sweeps.
├── ds.json                # Example dataset manifest (anonymized for public use).
├── requirements.txt       # Python dependency list.
├── README.md              # This README file.
├── LICENSE                # Project license.
└── ...                    # Other project files and directories.
```