# UCSF-PDGM Glioma Segmentation

A comprehensive pipeline for training, evaluating, and visualizing glioma segmentation models using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on the UCSF-PDGM dataset. This project focuses on robust evaluation metrics (Dice scores, inter-fold agreement) and high-quality qualitative visualizations for research reporting.

## Features

*   **Automated Dataset Preparation**: Converts UCSF-PDGM data into nnU-Net compliant format (`build_nnunet_dataset.py`).
*   **Comprehensive Evaluation**: Calculates Dice scores for all tumor regions (WT, TC, ET, ED, NCR) across individual folds and ensemble models (`evaluate_test_set.py`).
*   **Inter-Fold Agreement**: Analyzes consistency between different cross-validation folds.
*   **Visualization**: Generates publication-ready figures showing multi-modal MRI inputs and segmentation overlays (`create_qualitative_figure.py`).

## File Structure

```
├── build_nnunet_dataset.py      # Prepares raw naming convention for nnU-Net
├── evaluate_test_set.py         # Main evaluation script for Dice scores & stats
├── create_qualitative_figure.py # Generates visual comparisons of predictions
├── visualize_results.py         # Additional visualization utilities
├── patients_paths.csv           # Source mapping of patient files
├── nnUNet_raw/                  # Standard nnU-Net input directory
├── nnUNet_results/              # Training outputs and model weights
└── evaluation_results/          # CSV reports and summary statistics
```

## Setup & Installation

1.  **Environment**: Ensure you have a Python environment (e.g., venv or conda) with necessary dependencies:
    ```bash
    pip install nnunetv2 nibabel pandas numpy scipy matplotlib tqdm
    ```
    *(Note: Refer to official nnU-Net installation guide for full requirements)*

2.  **Dataset Preparation**:
    Map your local UCSF-PDGM file paths in `patients_paths.csv` and run:
    ```bash
    python build_nnunet_dataset.py --all
    ```
    This organizes data into `nnUNet_raw/Dataset001_UCSF`.

## Usage

### 1. Training & Inference
Run standard nnU-Net training and inference commands. Typically:
```bash
# Example for training fold 0
nnUNetv2_train 1 3d_fullres 0

# Example for inference
nnUNetv2_predict -i <input_folder> -o <output_folder> -d 1 -c 3d_fullres -f 0
```

### 2. Evaluation
To compute Dice scores and generate statistical reports:
```bash
python evaluate_test_set.py
```
Outputs will be saved to `evaluation_results/`:
*   `per_case_dice_scores.csv`: detailed metrics for every case.
*   `summary_statistics.csv`: mean/std/median/CI for each model and region.
*   `inter_fold_agreement.csv`: consistency analysis between folds.

### 3. Visualization
To create qualitative comparison figures:
```bash
python create_qualitative_figure.py
```

## Tumor Region Definitions (BraTS Convention)

*   **NCR (Necrotic Core)**: Label 1
*   **ED (Edema)**: Label 2
*   **ET (Enhancing Tumor)**: Label 3
*   **TC (Tumor Core)**: NCR + ET
*   **WT (Whole Tumor)**: NCR + ED + ET

## Results

Evaluation results are automatically aggregated in `evaluation_results/`. Check `summary_statistics.csv` for high-level model performance metrics.
