"""
Script to prepare exact file paths for each patient and modality in the UPENN dataset.

This script:
1. Scans the dataset directory
2. Identifies the exact file path for each modality for every patient
3. Saves the paths to patients_paths.csv
"""

import os
import pandas as pd
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Setup base paths for dataset and output."""
    base_path = Path(r"\\DXP4800PLUS-B85\UPENN dataset")
    data_dir = base_path / "UCSF-PDGM-v5"
    output_file = Path(__file__).parent / "patients_paths.csv"
    
    return data_dir, output_file


def get_patient_file_paths(data_dir):
    """
    Get exact file paths for all modalities for all patients.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with patient IDs and file paths for each modality
    """
    logger.info("=" * 80)
    logger.info("SCANNING FOR PATIENT FILE PATHS")
    logger.info("=" * 80)
    
    # All modality naming patterns
    modality_patterns = {
        'T1': ['_T1.nii', '_T1_'],
        'T1_bias': ['_T1_bias'],
        'T2': ['_T2.nii', '_T2_'],
        'T2_bias': ['_T2_bias'],
        'FLAIR': ['_FLAIR.nii', '_FLAIR_'],
        'FLAIR_bias': ['_FLAIR_bias'],
        'T1CE': ['_T1c.nii', '_T1c_'],  # T1 contrast-enhanced
        'T1CE_bias': ['_T1c_bias'],
        'ADC': ['_ADC.nii', '_ADC_'],
        'ASL': ['_ASL.nii', '_ASL_'],
        'DWI': ['_DWI.nii', '_DWI_'],
        'DWI_bias': ['_DWI_bias'],
        'SWI': ['_SWI.nii', '_SWI_'],
        'SWI_bias': ['_SWI_bias'],
        'DTI_FA': ['_DTI_eddy_FA'],
        'DTI_L1': ['_DTI_eddy_L1'],
        'DTI_L2': ['_DTI_eddy_L2'],
        'DTI_L3': ['_DTI_eddy_L3'],
        'DTI_MD': ['_DTI_eddy_MD'],
        'DTI_noreg': ['_DTI_eddy_noreg'],
        'brain_segmentation': ['_brain_segmentation'],
        'brain_parenchyma_segmentation': ['_brain_parenchyma_segmentation'],
        'tumor_segmentation': ['_tumor_segmentation'],
        'DTI_bvecs': ['_DTI_eddy,eddy_rotated_bvecs'],
    }
    
    patient_paths_data = []
    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    logger.info(f"\nScanning {len(patient_folders)} patient directories...")
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        
        # Initialize dictionary with patient ID
        patient_info = {'patient_id': patient_id}
        
        # Find all NIfTI files (and bvecs)
        all_files = list(patient_dir.glob('*'))
        
        # Track which modalities are found to avoid duplicates or ambiguity
        found_modalities = set()
        
        for file_path in all_files:
            filename = file_path.name
            
            # Check which modality this file represents
            for modality, patterns in modality_patterns.items():
                # Skip if we already found this modality for this patient
                if modality in found_modalities:
                    continue
                    
                for pattern in patterns:
                    if pattern in filename:
                        # Store the absolute path
                        patient_info[f'path_{modality}'] = str(file_path)
                        found_modalities.add(modality)
                        break
        
        patient_paths_data.append(patient_info)
    
    # Create DataFrame
    df = pd.DataFrame(patient_paths_data)
    
    # Fill NaN values with empty string or keep as NaN depending on preference
    # Keeping as NaN is usually better for analysis
    
    logger.info(f"\nGenerated paths for {len(df)} patients")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("PREPARE PATHS SCRIPT - STARTING")
        logger.info("=" * 80)
        
        # Step 1: Setup paths
        data_dir, output_file = setup_paths()
        logger.info(f"\nData directory: {data_dir}")
        logger.info(f"Output file: {output_file}")
        
        # Step 2: Get patient file paths
        df_paths = get_patient_file_paths(data_dir)
        
        # Step 3: Save to CSV
        logger.info(f"\nSaving paths to: {output_file}")
        df_paths.to_csv(output_file, index=False)
        logger.info("Successfully saved paths CSV")
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPARE PATHS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\nScript failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
