"""
Script to verify the integrity of the built nnU-Net dataset.
"""

import os
from pathlib import Path
import logging

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

def verify_dataset():
    logger.info(f"Verifying dataset in {DATASET_DIR}...")
    
    if not IMAGES_TR.exists() or not LABELS_TR.exists():
        logger.error("Images or Labels directory missing!")
        return

    # Get all unique case IDs from images
    image_files = list(IMAGES_TR.glob("*.nii.gz"))
    case_ids = set()
    for f in image_files:
        # Extract Case_XXXX from Case_XXXX_0000.nii.gz
        name_parts = f.name.split('_')
        # Assuming format UCSF-PDGM-XXXX_CHANNEL.nii.gz
        # We need to be careful with the splitting.
        # The build script uses: f"{clean_id}_{channel}.nii.gz"
        # clean_id is UCSF-PDGM-XXXX
        
        # Check if it ends with a channel suffix
        if f.name[-12:] in ['_0000.nii.gz', '_0001.nii.gz', '_0002.nii.gz', '_0003.nii.gz']:
            case_id = f.name[:-12]
            case_ids.add(case_id)
    
    logger.info(f"Found {len(case_ids)} unique cases based on images.")
    
    # Check for completeness
    issues = []
    
    for case_id in case_ids:
        # Check all 4 modalities
        for channel in ['0000', '0001', '0002', '0003']:
            img_path = IMAGES_TR / f"{case_id}_{channel}.nii.gz"
            if not img_path.exists():
                issues.append(f"Missing image: {img_path.name}")
        
        # Check label
        label_path = LABELS_TR / f"{case_id}.nii.gz"
        if not label_path.exists():
            issues.append(f"Missing label: {label_path.name}")
            
    if issues:
        logger.warning(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues)-10} more.")
    else:
        logger.info("✅ Dataset verification passed! All cases have 4 modalities and a label.")

if __name__ == "__main__":
    verify_dataset()
