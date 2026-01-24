"""
Visualize Ground Truth vs Predicted Segmentation (Side-by-Side)
Displays middle axial, sagittal, and coronal slices
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
INPUT_DIR = Path(r"c:\Khadiza\Segmentation_Project\test_input")
OUTPUT_DIR = Path(r"c:\Khadiza\Segmentation_Project\test_output")
LABELS_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\labelsTr")

def find_case_id():
    """Find the case ID from input folder"""
    files = list(INPUT_DIR.glob("*_0000.nii.gz"))
    if not files:
        print("ERROR: No input files found in test_input folder!")
        return None
    
    case_id = files[0].name.replace("_0000.nii.gz", "")
    return case_id

def load_and_visualize(case_id):
    """Load and visualize ground truth vs prediction"""
    
    # Load ground truth (if exists)
    gt_path = LABELS_DIR / f"{case_id}.nii.gz"
    pred_path = OUTPUT_DIR / f"{case_id}.nii.gz"
    
    if not pred_path.exists():
        print(f"ERROR: Prediction not found at {pred_path}")
        print("Run predict_test_subject.bat first!")
        return
    
    # Load prediction
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()
    
    # Load ground truth if available
    has_gt = gt_path.exists()
    if has_gt:
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata()
        print(f"Loaded ground truth from: {gt_path}")
    else:
        print(f"No ground truth found (testing on new data)")
        gt_data = None
    
    # Get middle slices
    mid_axial = pred_data.shape[2] // 2
    mid_sagittal = pred_data.shape[0] // 2
    mid_coronal = pred_data.shape[1] // 2
    
    # Create figure
    if has_gt:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.suptitle(f'Case: {case_id}', fontsize=16, fontweight='bold')
        
        # Axial view
        axes[0, 0].imshow(gt_data[:, :, mid_axial].T, cmap='jet', origin='lower')
        axes[0, 0].set_title('Ground Truth - Axial', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred_data[:, :, mid_axial].T, cmap='jet', origin='lower')
        axes[0, 1].set_title('Prediction - Axial', fontsize=12)
        axes[0, 1].axis('off')
        
        # Sagittal view
        axes[1, 0].imshow(gt_data[mid_sagittal, :, :].T, cmap='jet', origin='lower')
        axes[1, 0].set_title('Ground Truth - Sagittal', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_data[mid_sagittal, :, :].T, cmap='jet', origin='lower')
        axes[1, 1].set_title('Prediction - Sagittal', fontsize=12)
        axes[1, 1].axis('off')
        
        # Coronal view
        axes[2, 0].imshow(gt_data[:, mid_coronal, :].T, cmap='jet', origin='lower')
        axes[2, 0].set_title('Ground Truth - Coronal', fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(pred_data[:, mid_coronal, :].T, cmap='jet', origin='lower')
        axes[2, 1].set_title('Prediction - Coronal', fontsize=12)
        axes[2, 1].axis('off')
        
    else:
        # Only prediction (no ground truth)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Prediction: {case_id}', fontsize=16, fontweight='bold')
        
        axes[0].imshow(pred_data[:, :, mid_axial].T, cmap='jet', origin='lower')
        axes[0].set_title('Axial', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(pred_data[mid_sagittal, :, :].T, cmap='jet', origin='lower')
        axes[1].set_title('Sagittal', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(pred_data[:, mid_coronal, :].T, cmap='jet', origin='lower')
        axes[2].set_title('Coronal', fontsize=12)
        axes[2].axis('off')
    
    # Add colorbar legend
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
    cbar.set_label('Label: 0=BG, 1=NCR, 2=ED, 3=ET', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Save figure
    output_image = OUTPUT_DIR / f"{case_id}_visualization.png"
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_image}")
    
    plt.show()
    
    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Background (0): {np.sum(pred_data == 0)} voxels")
    print(f"  NCR (1):        {np.sum(pred_data == 1)} voxels")
    print(f"  ED (2):         {np.sum(pred_data == 2)} voxels")
    print(f"  ET (3):         {np.sum(pred_data == 3)} voxels")
    
    if has_gt:
        # Compute Dice scores
        dice_scores = []
        for label in [1, 2, 3]:
            pred_mask = (pred_data == label)
            gt_mask = (gt_data == label)
            intersection = np.sum(pred_mask & gt_mask)
            dice = 2 * intersection / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
            dice_scores.append(dice)
        
        print(f"\nDice Scores:")
        print(f"  NCR (1): {dice_scores[0]:.4f}")
        print(f"  ED (2):  {dice_scores[1]:.4f}")
        print(f"  ET (3):  {dice_scores[2]:.4f}")
        print(f"  Mean:    {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    print("="*60)
    print("Fold 0 Model - Segmentation Visualization")
    print("="*60)
    
    case_id = find_case_id()
    if case_id:
        print(f"\nFound case: {case_id}")
        load_and_visualize(case_id)
    else:
        print("\nPlease add test files to: test_input/")
        print("Required format: CASE-ID_0000.nii.gz, CASE-ID_0001.nii.gz, etc.")
