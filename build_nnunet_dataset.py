"""
Script to build an nnU-Net compliant dataset from the UPENN patient paths.

This script:
1. Creates the nnUNet_raw directory structure
2. Iterates over patients in patients_paths.csv
3. Copies and renames images to standard nnU-Net format (Case_XXXX_0000.nii.gz)
4. Copies and renames labels to standard nnU-Net format (Case_XXXX.nii.gz)
5. Verifies label values

Usage:
    python build_nnunet_dataset.py --patient_id UCSF-PDGM-0004_nifti  # Run for single patient
    python build_nnunet_dataset.py --all                              # Run for all patients
"""

import os
import shutil
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_UCSF"
BASE_PATH = Path(r"c:\Khadiza\Segmentation_Project")
NNUNET_RAW = BASE_PATH / "nnUNet_raw"
DATASET_DIR = NNUNET_RAW / DATASET_NAME
IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

# Modality to Channel mapping (Standard BraTS protocol)
# nnU-Net requires files to end with _0000, _0001, etc.
MODALITY_MAPPING = {
    'FLAIR': '0000',
    'T1': '0001',
    'T1CE': '0002',
    'T2': '0003'
}

def setup_directories():
    """Create necessary directories."""
    for dir_path in [IMAGES_TR, LABELS_TR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def remap_label(nifti_path, output_path):
    """
    Load label, check values, and optionally remap.
    For BraTS/Tumor segmentation, usually:
    0: Background
    1: NCR/NET (Necrotic/Non-Enhancing Tumor)
    2: Edema
    4: ET (Enhancing Tumor) -> Remap to 3 if needed for contiguous labels
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        unique_values = np.unique(data)
        logger.info(f"  Label values found: {unique_values}")
        
        # Example remapping: If labels are 0, 1, 2, 4 -> Remap 4 to 3
        # This ensures contiguous labels 0, 1, 2, 3
        if 4 in unique_values and 3 not in unique_values:
            logger.info("  Remapping label value 4 to 3...")
            data[data == 4] = 3
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, output_path)
        else:
            # Just copy if no remapping needed
            shutil.copy2(nifti_path, output_path)
            
    except Exception as e:
        logger.error(f"Error processing label {nifti_path}: {e}")
        raise

def process_patient(row, patient_id):
    """Process a single patient."""
    logger.info(f"Processing patient: {patient_id}")
    
    # 1. Process Images
    for modality, channel in MODALITY_MAPPING.items():
        col_name = f'path_{modality}'
        if col_name not in row or pd.isna(row[col_name]):
            logger.warning(f"  Missing path for {modality} in patient {patient_id}")
            continue
            
        src_path = Path(row[col_name])
        if not src_path.exists():
            logger.warning(f"  File not found: {src_path}")
            continue
            
        # Destination filename: {PatientID}_{Channel}.nii.gz
        # Cleaning PatientID to be safe (removing _nifti suffix if present for cleaner ID)
        clean_id = patient_id.replace('_nifti', '')
        dest_name = f"{clean_id}_{channel}.nii.gz"
        dest_path = IMAGES_TR / dest_name
        
        logger.info(f"  Copying {modality} -> {dest_name}")
        shutil.copy2(src_path, dest_path)

    # 2. Process Label (Tumor Segmentation)
    label_col = 'path_tumor_segmentation'
    if label_col in row and not pd.isna(row[label_col]):
        src_path = Path(row[label_col])
        if src_path.exists():
            clean_id = patient_id.replace('_nifti', '')
            dest_name = f"{clean_id}.nii.gz"
            dest_path = LABELS_TR / dest_name
            
            logger.info(f"  Processing Label -> {dest_name}")
            remap_label(src_path, dest_path)
        else:
            logger.warning(f"  Label file not found: {src_path}")
    else:
        logger.warning(f"  No tumor segmentation path for {patient_id}")

def main():
    parser = argparse.ArgumentParser(description="Build nnU-Net dataset")
    parser.add_argument("--patient_id", type=str, help="Process a specific patient ID")
    parser.add_argument("--all", action="store_true", help="Process all patients")
    args = parser.parse_args()
    
    # Load paths
    paths_csv = BASE_PATH / "patients_paths.csv"
    if not paths_csv.exists():
        logger.error("patients_paths.csv not found!")
        return
        
    df = pd.read_csv(paths_csv)
    
    # Setup directories
    setup_directories()
    
    # Filter patients
    if args.patient_id:
        df_filtered = df[df['patient_id'] == args.patient_id]
        if len(df_filtered) == 0:
            logger.error(f"Patient {args.patient_id} not found in CSV!")
            return
    elif args.all:
        df_filtered = df
    else:
        logger.info("Please specify --patient_id or --all")
        return
    
    logger.info(f"Processing {len(df_filtered)} patients...")
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        process_patient(row, row['patient_id'])
        
    logger.info("Dataset build completed!")
    logger.info(f"Images located in: {IMAGES_TR}")
    logger.info(f"Labels located in: {LABELS_TR}")

if __name__ == "__main__":
    main()
