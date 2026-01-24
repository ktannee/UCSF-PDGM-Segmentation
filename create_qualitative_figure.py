"""
Create Qualitative Visualization Figure (Similar to Fig 4 in Ranjbar et al.)

This script creates a multi-panel figure showing:
- Original MRI slices (T1CE typically)
- Ground truth segmentation overlays
- Prediction segmentation overlays
- Side-by-side comparison

For multiple representative cases (best, median, worst Dice scores)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd

# Paths
IMAGES_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\imagesTs")
GT_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\labelsTs")
PRED_DIR = Path(r"c:\Khadiza\Segmentation_Project\test_predictions_ensemble")
RESULTS_CSV = Path(r"c:\Khadiza\Segmentation_Project\evaluation_results\per_case_dice_scores.csv")
OUTPUT_DIR = Path(r"c:\Khadiza\Segmentation_Project\evaluation_results")

# Color scheme for segmentation regions (BraTS standard)
COLORS = {
    0: [0, 0, 0, 0],           # Background (transparent)
    1: [255, 0, 0, 180],       # NCR - Red
    2: [0, 255, 0, 180],       # ED - Green
    3: [0, 0, 255, 180],       # ET - Blue
}

REGION_NAMES = {
    1: 'NCR',
    2: 'ED', 
    3: 'ET'
}

def load_slice(nifti_path, slice_axis=2, slice_idx=None):
    """Load a 2D slice from a 3D NIfTI volume"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Auto-select middle slice if not specified
    if slice_idx is None:
        slice_idx = data.shape[slice_axis] // 2
    
    # Extract slice based on axis
    if slice_axis == 0:
        slice_2d = data[slice_idx, :, :]
    elif slice_axis == 1:
        slice_2d = data[:, slice_idx, :]
    else:  # axis 2 (axial)
        slice_2d = data[:, :, slice_idx]
    
    return slice_2d

def find_tumor_slice(seg_path, axis=2):
    """Find the slice with maximum tumor area"""
    seg = nib.load(seg_path).get_fdata()
    
    tumor_areas = []
    for i in range(seg.shape[axis]):
        if axis == 0:
            slice_2d = seg[i, :, :]
        elif axis == 1:
            slice_2d = seg[:, i, :]
        else:
            slice_2d = seg[:, :, i]
        
        tumor_area = np.sum(slice_2d > 0)
        tumor_areas.append(tumor_area)
    
    # Return slice with maximum tumor
    return np.argmax(tumor_areas)

def create_colored_overlay(seg_slice):
    """Convert segmentation slice to RGBA colored overlay"""
    h, w = seg_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    for label, color in COLORS.items():
        mask = seg_slice == label
        overlay[mask] = color
    
    return overlay

def plot_case_comparison(case_id, slice_idx=None, modality='T1CE'):
    """Create comparison plot for a single case"""
    
    # Determine modality channel (T1CE = 0002)
    modality_map = {'FLAIR': '0000', 'T1': '0001', 'T1CE': '0002', 'T2': '0003'}
    channel = modality_map.get(modality, '0002')
    
    # Load data
    img_path = IMAGES_DIR / f"{case_id}_{channel}.nii.gz"
    gt_path = GT_DIR / f"{case_id}.nii.gz"
    pred_path = PRED_DIR / f"{case_id}.nii.gz"
    
    if not all([p.exists() for p in [img_path, gt_path, pred_path]]):
        print(f"Warning: Missing files for {case_id}")
        return None
    
    # Auto-select best slice if not provided
    if slice_idx is None:
        slice_idx = find_tumor_slice(gt_path)
    
    # Load slices
    img_slice = load_slice(img_path, slice_idx=slice_idx)
    gt_slice = np.round(load_slice(gt_path, slice_idx=slice_idx)).astype(np.uint8)
    pred_slice = np.round(load_slice(pred_path, slice_idx=slice_idx)).astype(np.uint8)
    
    # Create colored overlays
    gt_overlay = create_colored_overlay(gt_slice)
    pred_overlay = create_colored_overlay(pred_slice)
    
    return img_slice, gt_overlay, pred_overlay, slice_idx

def create_qualitative_figure(num_cases=6):
    """
    Create multi-case qualitative figure
    
    Shows best, median, and worst cases based on WT Dice score
    """
    
    # Load results
    results_df = pd.read_csv(RESULTS_CSV)
    ensemble_df = results_df[results_df['model'] == 'Ensemble'].copy()
    
    # Sort by WT Dice
    ensemble_df = ensemble_df.sort_values('WT', ascending=False)
    
    # Select cases: best, median, worst
    n = len(ensemble_df)
    indices = [0, n//4, n//2, 3*n//4, n-2, n-1]  # Best, Q1, median, Q3, 2nd worst, worst
    selected_cases = ensemble_df.iloc[indices[:num_cases]]
    
    # Create figure
    fig, axes = plt.subplots(num_cases, 3, figsize=(15, 5*num_cases))
    
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(selected_cases.iterrows()):
        case_id = row['case_id']
        
        # Plot case
        result = plot_case_comparison(case_id)
        
        if result is None:
            continue
        
        img_slice, gt_overlay, pred_overlay, slice_idx = result
        
        # Column 1: Original image
        axes[idx, 0].imshow(img_slice, cmap='gray')
        axes[idx, 0].set_title(f"{case_id}\nT1CE (Slice {slice_idx})", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Column 2: Ground Truth overlay
        axes[idx, 1].imshow(img_slice, cmap='gray')
        axes[idx, 1].imshow(gt_overlay, alpha=0.5)
        axes[idx, 1].set_title(f"Ground Truth\nWT Dice: {row['WT']:.3f}", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Column 3: Prediction overlay
        axes[idx, 2].imshow(img_slice, cmap='gray')
        axes[idx, 2].imshow(pred_overlay, alpha=0.5)
        dice_text = f"Prediction\nED:{row['ED']:.2f} ET:{row['ET']:.2f} NCR:{row['NCR']:.2f} TC:{row['TC']:.2f}"
        axes[idx, 2].set_title(dice_text, fontsize=10)
        axes[idx, 2].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(COLORS[1])/255, label='NCR (Necrotic Core)'),
        mpatches.Patch(facecolor=np.array(COLORS[2])/255, label='ED (Edema)'),
        mpatches.Patch(facecolor=np.array(COLORS[3])/255, label='ET (Enhancing Tumor)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    
    # Save figure
    output_path = OUTPUT_DIR / "qualitative_figure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved qualitative figure to: {output_path}")
    
    plt.close()
    
    return output_path

def create_single_case_figure(case_id, output_name="single_case_viz.png"):
    """Create detailed visualization for a single case"""
    
    result = plot_case_comparison(case_id)
    
    if result is None:
        print(f"Could not create visualization for {case_id}")
        return None
    
    img_slice, gt_overlay, pred_overlay, slice_idx = result
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f"{case_id} - T1CE (Slice {slice_idx})", fontsize=14)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title("Ground Truth Segmentation", fontsize=14)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_overlay, alpha=0.5)
    axes[2].set_title("Model Prediction", fontsize=14)
    axes[2].axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(COLORS[1])/255, label='NCR (Necrotic Core)'),
        mpatches.Patch(facecolor=np.array(COLORS[2])/255, label='ED (Edema)'),
        mpatches.Patch(facecolor=np.array(COLORS[3])/255, label='ET (Enhancing Tumor)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved single case figure to: {output_path}")
    
    plt.close()
    
    return output_path

if __name__ == "__main__":
    print("="*70)
    print("Creating Qualitative Visualization Figures")
    print("="*70)
    
    # Create multi-case figure (Fig 4 style)
    print("\nCreating multi-case comparison figure...")
    create_qualitative_figure(num_cases=6)
    
    # Optionally create single case examples
    print("\nCreating single case examples...")
    
    # Load results to find best case
    results_df = pd.read_csv(RESULTS_CSV)
    ensemble_df = results_df[results_df['model'] == 'Ensemble']
    
    # Best WT Dice case
    best_case = ensemble_df.loc[ensemble_df['WT'].idxmax(), 'case_id']
    create_single_case_figure(best_case, "best_case_visualization.png")
    
    # Median WT Dice case
    median_idx = len(ensemble_df) // 2
    median_case = ensemble_df.sort_values('WT').iloc[median_idx]['case_id']
    create_single_case_figure(median_case, "median_case_visualization.png")
    
    print("\n" + "="*70)
    print("Qualitative Visualization Complete!")
    print("="*70)
