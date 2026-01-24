"""
Script to select patients with all required modalities from UPENN dataset.

This script:
1. Inspects the metadata to understand patient information
2. Analyzes available imaging channels/modalities for each patient
3. Filters patients to only those with ALL required modalities
4. Saves the filtered patient list as patients_selected.csv
"""

import os
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Setup base paths for dataset and metadata."""
    base_path = Path(r"\\DXP4800PLUS-B85\UPENN dataset")
    data_dir = base_path / "UCSF-PDGM-v5"
    metadata_file = base_path / "UCSF-PDGM-metadata_v5"
    output_file = Path(__file__).parent / "patients_selected.csv"
    
    return data_dir, metadata_file, output_file


def find_metadata_file(metadata_path):
    """
    Find the metadata file with common extensions.
    
    Args:
        metadata_path: Base path without extension
        
    Returns:
        Path to the metadata file if found, None otherwise
    """
    common_extensions = ['.csv', '.xlsx', '.xls', '.tsv', '.txt', '.json']
    
    # Check if the path already has an extension
    if metadata_path.exists() and metadata_path.is_file():
        return metadata_path
    
    # Try common extensions
    for ext in common_extensions:
        file_path = Path(str(metadata_path) + ext)
        if file_path.exists():
            logger.info(f"Found metadata file: {file_path}")
            return file_path
    
    return None


def load_metadata(metadata_file):
    """
    Load metadata file and return as DataFrame.
    
    Args:
        metadata_file: Path to metadata file
        
    Returns:
        pandas DataFrame containing metadata
    """
    file_ext = metadata_file.suffix.lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(metadata_file)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(metadata_file)
        elif file_ext == '.tsv' or file_ext == '.txt':
            df = pd.read_csv(metadata_file, sep='\t')
        elif file_ext == '.json':
            df = pd.read_json(metadata_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Successfully loaded metadata with {len(df)} records")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        raise


def inspect_metadata(df):
    """
    Inspect metadata to understand the structure and content.
    
    Args:
        df: DataFrame containing metadata
    """
    logger.info("=" * 80)
    logger.info("METADATA INSPECTION")
    logger.info("=" * 80)
    
    # Basic info
    logger.info(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"\nColumn Names and Types:")
    for col in df.columns:
        logger.info(f"  - {col}: {df[col].dtype}")
    
    # Missing values
    logger.info(f"\nMissing Values:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        logger.info(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Sample rows
    logger.info(f"\nFirst 5 rows:")
    print(df.head())
    
    # Unique values for key columns
    logger.info(f"\nUnique Values Summary:")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 20:  # Only show for columns with few unique values
            logger.info(f"  - {col}: {unique_count} unique values")
            logger.info(f"    Values: {df[col].unique().tolist()}")
        else:
            logger.info(f"  - {col}: {unique_count} unique values")
    
    logger.info("=" * 80)


def detect_modalities_from_directory(data_dir):
    """
    Detect available modalities by inspecting patient directories.
    
    Args:
        data_dir: Path to data directory containing patient folders
        
    Returns:
        Dictionary with patient IDs as keys and list of modalities as values
    """
    logger.info("\nDetecting modalities from patient directories...")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}
    
    # Common MRI modality naming patterns
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
    }
    
    patient_modalities = {}
    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    logger.info(f"Found {len(patient_folders)} patient directories")
    
    for patient_dir in patient_folders[:10]:  # Inspect first 10 patients for speed
        patient_id = patient_dir.name
        modalities_found = set()
        
        # Look for NIfTI files in the patient directory
        nifti_files = list(patient_dir.glob('*.nii')) + list(patient_dir.glob('*.nii.gz'))
        
        for nifti_file in nifti_files:
            filename = nifti_file.stem.lower()
            if filename.endswith('.nii'):
                filename = filename[:-4]
            
            # Check which modality this file represents
            for modality, patterns in modality_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in filename:
                        modalities_found.add(modality)
                        break
        
        patient_modalities[patient_id] = list(modalities_found)
        
        if len(patient_modalities) == 1:
            logger.info(f"Sample patient {patient_id}: {modalities_found}")
    
    return patient_modalities, modality_patterns


def decide_required_channels(patient_modalities):
    """
    Decide which channels/modalities are required based on availability.
    
    Args:
        patient_modalities: Dictionary of patient IDs and their available modalities
        
    Returns:
        Set of required modalities
    """
    logger.info("\n" + "=" * 80)
    logger.info("CHANNEL/MODALITY DECISION")
    logger.info("=" * 80)
    
    # Count how many patients have each modality
    modality_counts = defaultdict(int)
    for patient_id, modalities in patient_modalities.items():
        for modality in modalities:
            modality_counts[modality] += 1
    
    total_patients = len(patient_modalities)
    
    logger.info("\nModality availability across patients:")
    for modality, count in sorted(modality_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_patients) * 100
        logger.info(f"  - {modality}: {count}/{total_patients} patients ({percentage:.1f}%)")
    
    # Standard brain tumor MRI modalities (BraTS-style)
    # Typically requires: T1, T1CE, T2, FLAIR
    standard_modalities = {'T1', 'T1CE', 'T2', 'FLAIR'}
    
    # Find which modalities are available in most patients
    common_modalities = {mod for mod, count in modality_counts.items() 
                        if count / total_patients >= 0.7}  # Present in at least 70% of patients
    
    # Decide on required modalities
    if standard_modalities.issubset(set(modality_counts.keys())):
        required_modalities = standard_modalities
        logger.info(f"\nUsing standard brain tumor MRI modalities: {required_modalities}")
    elif common_modalities:
        required_modalities = common_modalities
        logger.info(f"\nUsing commonly available modalities: {required_modalities}")
    else:
        # Use all available modalities
        required_modalities = set(modality_counts.keys())
        logger.info(f"\nUsing all detected modalities: {required_modalities}")
    
    logger.info("=" * 80)
    
    return required_modalities


def filter_patients_with_all_modalities(data_dir, required_modalities):
    """
    Filter patients that have all required modalities.
    
    Args:
        data_dir: Path to data directory
        required_modalities: Set of required modality names
        
    Returns:
        DataFrame with filtered patients and their modalities
    """
    logger.info("\n" + "=" * 80)
    logger.info("FILTERING PATIENTS BY MODALITIES")
    logger.info("=" * 80)
    logger.info(f"\nRequired modalities: {required_modalities}")
    
    # Common MRI modality naming patterns
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
    }
    
    patient_data = []
    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    logger.info(f"\nScanning {len(patient_folders)} patient directories...")
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        
        # Find all NIfTI files
        nifti_files = list(patient_dir.glob('*.nii')) + list(patient_dir.glob('*.nii.gz'))
        
        # Track which modalities are present
        modalities_present = set()
        modality_files = {}
        
        for nifti_file in nifti_files:
            filename = nifti_file.name.lower()
            
            # Check which modality this file represents
            for modality in required_modalities:
                if modality in modality_patterns:
                    patterns = modality_patterns[modality]
                    for pattern in patterns:
                        if pattern.lower() in filename:
                            modalities_present.add(modality)
                            modality_files[modality] = nifti_file.name
                            break
        
        # Check if patient has all required modalities
        has_all_modalities = required_modalities.issubset(modalities_present)
        
        patient_info = {
            'patient_id': patient_id,
            'has_all_modalities': has_all_modalities,
            'modalities_present': ', '.join(sorted(modalities_present)),
            'modality_count': len(modalities_present),
            'directory_path': str(patient_dir)
        }
        
        # Add individual modality flags
        for modality in required_modalities:
            patient_info[f'has_{modality}'] = modality in modalities_present
        
        patient_data.append(patient_info)
    
    # Create DataFrame
    df = pd.DataFrame(patient_data)
    
    # Filter to patients with all modalities
    df_filtered = df[df['has_all_modalities'] == True].copy()
    
    # Summary statistics
    logger.info(f"\nTotal patients scanned: {len(df)}")
    logger.info(f"Patients with all required modalities: {len(df_filtered)}")
    logger.info(f"Patients excluded: {len(df) - len(df_filtered)}")
    
    if len(df_filtered) > 0:
        logger.info(f"\nModality coverage in filtered dataset:")
        for modality in required_modalities:
            count = df_filtered[f'has_{modality}'].sum()
            logger.info(f"  - {modality}: {count}/{len(df_filtered)} patients (100%)")
    
    # Show some excluded patients
    df_excluded = df[df['has_all_modalities'] == False]
    if len(df_excluded) > 0:
        logger.info(f"\nSample of excluded patients (first 5):")
        for idx, row in df_excluded.head().iterrows():
            missing = required_modalities - set(row['modalities_present'].split(', '))
            logger.info(f"  - {row['patient_id']}: Missing {missing}")
    
    logger.info("=" * 80)
    
    return df_filtered, df


def collect_all_patient_modalities(data_dir):
    """
    Collect ALL available modalities for ALL patients.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with all patients and their available modalities
    """
    logger.info("\n" + "=" * 80)
    logger.info("COLLECTING ALL PATIENT MODALITIES")
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
    
    patient_data = []
    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    logger.info(f"\nScanning {len(patient_folders)} patient directories...")
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        
        # Find all NIfTI files
        nifti_files = list(patient_dir.glob('*.nii')) + list(patient_dir.glob('*.nii.gz'))
        
        # Track which modalities are present
        modalities_present = set()
        modality_files = {}
        
        for nifti_file in nifti_files:
            filename = nifti_file.name
            
            # Check which modality this file represents
            for modality, patterns in modality_patterns.items():
                for pattern in patterns:
                    if pattern in filename:
                        modalities_present.add(modality)
                        modality_files[modality] = nifti_file.name
                        break
        
        patient_info = {
            'patient_id': patient_id,
            'modalities_present': ', '.join(sorted(modalities_present)),
            'modality_count': len(modalities_present),
            'directory_path': str(patient_dir)
        }
        
        # Add individual modality flags for all possible modalities
        for modality in modality_patterns.keys():
            patient_info[f'has_{modality}'] = modality in modalities_present
        
        patient_data.append(patient_info)
    
    # Create DataFrame
    df = pd.DataFrame(patient_data)
    
    # Summary statistics
    logger.info(f"\nTotal patients scanned: {len(df)}")
    logger.info(f"\nModality availability across all patients:")
    for modality in sorted(modality_patterns.keys()):
        if f'has_{modality}' in df.columns:
            count = df[f'has_{modality}'].sum()
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"  - {modality}: {count}/{len(df)} patients ({percentage:.1f}%)")
    
    logger.info("=" * 80)
    
    return df


def save_patient_list(df_filtered, output_file):
    """
    Save the filtered patient list to CSV file.
    
    Args:
        df_filtered: DataFrame with filtered patients
        output_file: Path to output CSV file
    """
    logger.info(f"\nSaving patient list to: {output_file}")
    
    df_filtered.to_csv(output_file, index=False)
    
    logger.info(f"Successfully saved {len(df_filtered)} patients to CSV file")
    
    # Also save a summary JSON
    summary_file = output_file.parent / "patients_selected_summary.json"
    summary = {
        'total_patients_selected': len(df_filtered),
        'selection_criteria': 'All required modalities present',
        'patient_ids': df_filtered['patient_id'].tolist()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("PATIENT SELECTION SCRIPT - STARTING")
        logger.info("=" * 80)
        
        # Step 1: Setup paths
        data_dir, metadata_path, output_file = setup_paths()
        logger.info(f"\nData directory: {data_dir}")
        logger.info(f"Output file: {output_file}")
        
        # Step 2: Find and load metadata
        metadata_file = find_metadata_file(metadata_path)
        if metadata_file is not None:
            logger.info(f"Metadata file: {metadata_file}")
            df_metadata = load_metadata(metadata_file)
            
            # Inspect metadata
            inspect_metadata(df_metadata)
        else:
            logger.warning("Metadata file not found - proceeding with directory inspection only")
            df_metadata = None
        
        # Step 3: Detect modalities from a sample of patient directories
        patient_modalities, modality_patterns = detect_modalities_from_directory(data_dir)
        
        # Step 4: Decide required channels/modalities
        if patient_modalities:
            required_modalities = decide_required_channels(patient_modalities)
        else:
            # Default to standard brain tumor modalities
            required_modalities = {'T1', 'T1CE', 'T2', 'FLAIR'}
            logger.info(f"\nUsing default required modalities: {required_modalities}")
        
        # Step 5: Collect ALL modalities for ALL patients (not just filtering)
        df_all = collect_all_patient_modalities(data_dir)
        
        # Save all patient data with their modalities
        save_patient_list(df_all, output_file)
        
        logger.info("\n" + "=" * 80)
        logger.info("PATIENT SELECTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\nScript failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
