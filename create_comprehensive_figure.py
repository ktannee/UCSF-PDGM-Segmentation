"""
Create Comprehensive Qualitative Visualization Figure

Shows all 4 MRI modalities (FLAIR, T1, T1CE, T2) with predictions from all models
(Ensemble + 5 folds) for 5 random test cases.

Layout:
- Rows: 5 random test cases
- Columns: 4 modalities × (GT + 6 models) = 28 columns per row
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd
import random

# Paths
IMAGES_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\imagesTs")
GT_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\labelsTs")
ENSEMBLE_PRED_DIR = Path(r"c:\Khadiza\Segmentation_Project\test_predictions_ensemble")
FOLD_PRED_DIRS = {
    0: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold0"),
    1: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold1"),
    2: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold2"),
    3: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold3"),
    4: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold4"),
}
RESULTS_CSV = Path(r"c:\Khadiza\Segmentation_Project\evaluation_results\per_case_dice_scores.csv")
OUTPUT_DIR = Path(r"c:\Khadiza\Segmentation_Project\evaluation_results")

# Modality mapping
MODALITIES = {
    'FLAIR': '0000',
    'T1': '0001',
    'T1CE': '0002',
    'T2': '0003'
}

# Color scheme for segmentation regions (BraTS standard)
COLORS = {
    0: [0, 0, 0, 0],           # Background (transparent)
    1: [255, 0, 0, 180],       # NCR - Red
    2: [0, 255, 0, 180],       # ED - Green
    3: [0, 0, 255, 180],       # ET - Blue
}

def load_slice(nifti_path, slice_axis=2, slice_idx=None):
    """Load a 2D slice from a 3D NIfTI volume"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    if slice_idx is None:
        slice_idx = data.shape[slice_axis] // 2
    
    if slice_axis == 0:
        slice_2d = data[slice_idx, :, :]
    elif slice_axis == 1:
        slice_2d = data[:, slice_idx, :]
    else:
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
    
    return np.argmax(tumor_areas)

def create_colored_overlay(seg_slice):
    """Convert segmentation slice to RGBA colored overlay"""
    h, w = seg_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    for label, color in COLORS.items():
        mask = seg_slice == label
        overlay[mask] = color
    
    return overlay

def create_comprehensive_figure(num_cases=5, random_seed=42):
    """
    Create comprehensive figure showing all modalities and all models
    
    Args:
        num_cases: Number of random test cases to show
        random_seed: Random seed for reproducibility
    """
    
    # Load results
    results_df = pd.read_csv(RESULTS_CSV)
    ensemble_df = results_df[results_df['model'] == 'Ensemble'].copy()
    
    # Select random cases
    random.seed(random_seed)
    all_cases = ensemble_df['case_id'].tolist()
    selected_cases = random.sample(all_cases, min(num_cases, len(all_cases)))
    
    print(f"Selected cases: {selected_cases}")
    
    # Create figure with subplots
    # Layout: num_cases rows × (4 modalities × 7 model predictions) columns
    # Each modality shows: Original + GT + Ensemble + 5 Folds
    n_cols_per_modality = 7  # Original + GT + Ensemble + 5 Folds
    n_cols = len(MODALITIES) * n_cols_per_modality
    
    fig, axes = plt.subplots(num_cases, n_cols, figsize=(3.5*n_cols, 4*num_cases))
    
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    for case_idx, case_id in enumerate(selected_cases):
        print(f"Processing {case_id}...")
        
        # Find best slice
        gt_path = GT_DIR / f"{case_id}.nii.gz"
        slice_idx = find_tumor_slice(gt_path)
        
        # Load ground truth
        gt_slice = np.round(load_slice(gt_path, slice_idx=slice_idx)).astype(np.uint8)
        gt_overlay = create_colored_overlay(gt_slice)
        
        # Load predictions
        ensemble_path = ENSEMBLE_PRED_DIR / f"{case_id}.nii.gz"
        ensemble_slice = np.round(load_slice(ensemble_path, slice_idx=slice_idx)).astype(np.uint8)
        ensemble_overlay = create_colored_overlay(ensemble_slice)
        
        fold_overlays = []
        for fold in range(5):
            fold_path = FOLD_PRED_DIRS[fold] / f"{case_id}.nii.gz"
            fold_slice = np.round(load_slice(fold_path, slice_idx=slice_idx)).astype(np.uint8)
            fold_overlays.append(create_colored_overlay(fold_slice))
        
        # Get Dice scores for this case
        case_results = ensemble_df[ensemble_df['case_id'] == case_id].iloc[0]
        
        # Plot each modality
        col_offset = 0
        for mod_idx, (mod_name, mod_channel) in enumerate(MODALITIES.items()):
            # Load modality image
            img_path = IMAGES_DIR / f"{case_id}_{mod_channel}.nii.gz"
            img_slice = load_slice(img_path, slice_idx=slice_idx)
            
            # Column 0: Original image
            ax = axes[case_idx, col_offset]
            ax.imshow(img_slice, cmap='gray')
            if case_idx == 0:
                ax.set_title(f"{mod_name}\nOriginal", fontsize=8, fontweight='bold')
            if mod_idx == 0:
                ax.set_ylabel(f"{case_id}\nSlice {slice_idx}", fontsize=8, fontweight='bold')
            ax.axis('off')
            
            # Column 1: Ground Truth
            ax = axes[case_idx, col_offset + 1]
            ax.imshow(img_slice, cmap='gray')
            ax.imshow(gt_overlay, alpha=0.5)
            if case_idx == 0:
                ax.set_title(f"{mod_name}\nGround Truth", fontsize=8, fontweight='bold')
            ax.axis('off')
            
            # Column 2: Ensemble
            ax = axes[case_idx, col_offset + 2]
            ax.imshow(img_slice, cmap='gray')
            ax.imshow(ensemble_overlay, alpha=0.5)
            if case_idx == 0:
                ax.set_title(f"{mod_name}\nEnsemble", fontsize=8, fontweight='bold')
            ax.axis('off')
            
            # Columns 3-7: Individual folds
            for fold in range(5):
                ax = axes[case_idx, col_offset + 3 + fold]
                ax.imshow(img_slice, cmap='gray')
                ax.imshow(fold_overlays[fold], alpha=0.5)
                if case_idx == 0:
                    ax.set_title(f"{mod_name}\nFold {fold}", fontsize=8, fontweight='bold')
                ax.axis('off')
            
            col_offset += n_cols_per_modality
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(COLORS[1])/255, label='NCR (Necrotic Core)'),
        mpatches.Patch(facecolor=np.array(COLORS[2])/255, label='ED (Edema)'),
        mpatches.Patch(facecolor=np.array(COLORS[3])/255, label='ET (Enhancing Tumor)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=True)
    
    # Add title
    fig.suptitle(f'Comprehensive Multi-Modal Multi-Model Segmentation Results\n'
                 f'Showing {num_cases} Random Test Cases with All 4 MRI Modalities and All Model Predictions',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, bottom=0.02, hspace=0.1, wspace=0.05)
    
    # Save figure
    output_path = OUTPUT_DIR / "comprehensive_multimodal_figure.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive figure to: {output_path}")
    
    plt.close()
    
    return output_path

def create_compact_figure(num_cases=5, random_seed=42):
    """
    Create a more compact version showing only key comparisons
    
    Layout per case:
    - Row 1: All 4 modalities (original images)
    - Row 2: All 4 modalities with GT overlay
    - Row 3: All 4 modalities with Ensemble prediction overlay
    """
    
    # Load results
    results_df = pd.read_csv(RESULTS_CSV)
    ensemble_df = results_df[results_df['model'] == 'Ensemble'].copy()
    
    # Select random cases
    random.seed(random_seed)
    all_cases = ensemble_df['case_id'].tolist()
    selected_cases = random.sample(all_cases, min(num_cases, len(all_cases)))
    
    print(f"\nCreating compact figure for cases: {selected_cases}")
    
    # Create figure: (num_cases × 3 rows) × 4 columns
    fig, axes = plt.subplots(num_cases * 3, 4, figsize=(16, 4*num_cases*3))
    
    for case_idx, case_id in enumerate(selected_cases):
        print(f"Processing {case_id}...")
        
        # Find best slice
        gt_path = GT_DIR / f"{case_id}.nii.gz"
        slice_idx = find_tumor_slice(gt_path)
        
        # Load ground truth and prediction
        gt_slice = np.round(load_slice(gt_path, slice_idx=slice_idx)).astype(np.uint8)
        gt_overlay = create_colored_overlay(gt_slice)
        
        ensemble_path = ENSEMBLE_PRED_DIR / f"{case_id}.nii.gz"
        ensemble_slice = np.round(load_slice(ensemble_path, slice_idx=slice_idx)).astype(np.uint8)
        ensemble_overlay = create_colored_overlay(ensemble_slice)
        
        # Get Dice scores
        case_results = ensemble_df[ensemble_df['case_id'] == case_id].iloc[0]
        dice_text = f"WT:{case_results['WT']:.3f} TC:{case_results['TC']:.3f} ET:{case_results['ET']:.3f}"
        
        base_row = case_idx * 3
        
        # Plot each modality
        for mod_idx, (mod_name, mod_channel) in enumerate(MODALITIES.items()):
            # Load modality image
            img_path = IMAGES_DIR / f"{case_id}_{mod_channel}.nii.gz"
            img_slice = load_slice(img_path, slice_idx=slice_idx)
            
            # Row 1: Original image
            ax = axes[base_row, mod_idx]
            ax.imshow(img_slice, cmap='gray')
            if mod_idx == 0:
                ax.set_ylabel(f"{case_id}\nOriginal", fontsize=10, fontweight='bold')
            if case_idx == 0:
                ax.set_title(mod_name, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Row 2: Ground Truth overlay
            ax = axes[base_row + 1, mod_idx]
            ax.imshow(img_slice, cmap='gray')
            ax.imshow(gt_overlay, alpha=0.5)
            if mod_idx == 0:
                ax.set_ylabel("Ground Truth", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Row 3: Ensemble prediction overlay
            ax = axes[base_row + 2, mod_idx]
            ax.imshow(img_slice, cmap='gray')
            ax.imshow(ensemble_overlay, alpha=0.5)
            if mod_idx == 0:
                ax.set_ylabel(f"Prediction\n{dice_text}", fontsize=9, fontweight='bold')
            ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=np.array(COLORS[1])/255, label='NCR (Necrotic Core)'),
        mpatches.Patch(facecolor=np.array(COLORS[2])/255, label='ED (Edema)'),
        mpatches.Patch(facecolor=np.array(COLORS[3])/255, label='ET (Enhancing Tumor)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.01), fontsize=12, frameon=True)
    
    fig.suptitle(f'Multi-Modal Segmentation Results - Ensemble Model\n'
                 f'{num_cases} Random Test Cases',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, bottom=0.02, hspace=0.05, wspace=0.05)
    
    # Save figure
    output_path = OUTPUT_DIR / "compact_multimodal_figure.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved compact figure to: {output_path}")
    
    plt.close()
    
    return output_path

if __name__ == "__main__":
    print("="*70)
    print("Creating Comprehensive Multi-Modal Multi-Model Figures")
    print("="*70)
    
    # Create compact version (recommended for papers)
    print("\n[1/2] Creating compact multi-modal figure (Ensemble only)...")
    create_compact_figure(num_cases=5, random_seed=42)
    
    # Create comprehensive version (all models)
    print("\n[2/2] Creating comprehensive figure (All models)...")
    print("Warning: This will be a very large figure!")
    create_comprehensive_figure(num_cases=5, random_seed=42)
    
    print("\n" + "="*70)
    print("Multi-Modal Visualization Complete!")
    print("="*70)
