"""
Comprehensive Test Set Evaluation Script
Computes Dice scores for 5 regions across ensemble and individual folds
Generates statistics, agreement analysis, and result summaries
"""
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# Paths
GROUND_TRUTH_DIR = Path(r"c:\Khadiza\Segmentation_Project\nnUNet_raw\Dataset001_UCSF\labelsTs")
ENSEMBLE_PRED_DIR = Path(r"c:\Khadiza\Segmentation_Project\test_predictions_ensemble")
FOLD_PRED_DIRS = {
    0: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold0"),
    1: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold1"),
    2: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold2"),
    3: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold3"),
    4: Path(r"c:\Khadiza\Segmentation_Project\test_predictions_fold4"),
}
OUTPUT_DIR = Path(r"c:\Khadiza\Segmentation_Project\evaluation_results")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Region definitions (BraTS convention)
REGIONS = {
    'ED': 'Edema',
    'ET': 'Enhancing Tumor',
    'NCR': 'Necrotic Core',
    'TC': 'Tumor Core',
    'WT': 'Whole Tumor'
}

def compute_region_mask(segmentation, region):
    """
    Extract binary mask for a specific region
    
    BraTS label convention:
    - 0: Background
    - 1: NCR (Necrotic Core)
    - 2: ED (Edema)
    - 3: ET (Enhancing Tumor)
    """
    if region == 'ED':
        return (segmentation == 2).astype(np.uint8)
    elif region == 'ET':  # Enhancing Tumor
        return (segmentation == 3).astype(np.uint8)
    elif region == 'NCR':  # Necrotic Core
        return (segmentation == 1).astype(np.uint8)
    elif region == 'TC':  # Tumor Core = NCR + ET
        return ((segmentation == 1) | (segmentation == 3)).astype(np.uint8)
    elif region == 'WT':  # Whole Tumor
        return (segmentation > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown region: {region}")

def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient"""
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / union
    return dice

def evaluate_case(case_id, pred_dir, gt_dir):
    """Evaluate all regions for a single case"""
    pred_path = pred_dir / f"{case_id}.nii.gz"
    gt_path = gt_dir / f"{case_id}.nii.gz"
    
    if not pred_path.exists():
        print(f"Warning: Prediction not found for {case_id}")
        return None
    
    if not gt_path.exists():
        print(f"Warning: Ground truth not found for {case_id}")
        return None
    
    # Load data and convert to integers to avoid float comparison issues
    pred_data = np.round(nib.load(pred_path).get_fdata()).astype(np.uint8)
    gt_data = np.round(nib.load(gt_path).get_fdata()).astype(np.uint8)
    
    # Compute Dice for all regions
    results = {'case_id': case_id}
    
    for region in REGIONS.keys():
        pred_mask = compute_region_mask(pred_data, region)
        gt_mask = compute_region_mask(gt_data, region)
        dice = compute_dice(pred_mask, gt_mask)
        results[region] = dice
    
    return results

def evaluate_all_cases(pred_dir, model_name):
    """Evaluate all test cases for a specific model"""
    print(f"\nEvaluating {model_name}...")
    
    # Get all test cases
    gt_files = sorted(GROUND_TRUTH_DIR.glob("*.nii.gz"))
    case_ids = [f.stem.replace('.nii', '') for f in gt_files]  # Handle .nii.gz double extension
    
    print(f"Found {len(case_ids)} test cases")
    
    results = []
    for case_id in tqdm(case_ids, desc=f"Processing {model_name}"):
        case_result = evaluate_case(case_id, pred_dir, GROUND_TRUTH_DIR)
        if case_result:
            case_result['model'] = model_name
            results.append(case_result)
    
    return pd.DataFrame(results)

def compute_statistics(df, model_name):
    """Compute summary statistics for a model"""
    stats_dict = {
        'model': model_name,
        'n_cases': len(df)
    }
    
    for region in REGIONS.keys():
        dice_values = df[region].values
        stats_dict[f'{region}_mean'] = np.mean(dice_values)
        stats_dict[f'{region}_std'] = np.std(dice_values)
        stats_dict[f'{region}_median'] = np.median(dice_values)
        stats_dict[f'{region}_min'] = np.min(dice_values)
        stats_dict[f'{region}_max'] = np.max(dice_values)
        
        # 95% Confidence Interval
        ci = stats.t.interval(0.95, len(dice_values)-1, 
                              loc=np.mean(dice_values), 
                              scale=stats.sem(dice_values))
        stats_dict[f'{region}_ci_lower'] = ci[0]
        stats_dict[f'{region}_ci_upper'] = ci[1]
    
    return stats_dict

def compute_inter_fold_agreement(all_results_df):
    """Compute agreement metrics between folds"""
    print("\nComputing inter-fold agreement...")
    
    # Get fold results
    fold_dfs = {}
    for fold in range(5):
        fold_df = all_results_df[all_results_df['model'] == f'Fold_{fold}']
        fold_dfs[fold] = fold_df.set_index('case_id')
    
    agreement_results = []
    
    # For each case
    case_ids = fold_dfs[0].index
    for case_id in case_ids:
        case_agreement = {'case_id': case_id}
        
        for region in REGIONS.keys():
            # Get Dice scores from all folds
            fold_dices = [fold_dfs[fold].loc[case_id, region] for fold in range(5)]
            
            # Compute statistics
            case_agreement[f'{region}_mean'] = np.mean(fold_dices)
            case_agreement[f'{region}_std'] = np.std(fold_dices)
            case_agreement[f'{region}_range'] = np.max(fold_dices) - np.min(fold_dices)
            
            # Agreement within 0.05
            max_diff = np.max(fold_dices) - np.min(fold_dices)
            case_agreement[f'{region}_agreement_0.05'] = max_diff < 0.05
        
        agreement_results.append(case_agreement)
    
    agreement_df = pd.DataFrame(agreement_results)
    
    # Summary statistics
    agreement_summary = {}
    for region in REGIONS.keys():
        agreement_summary[f'{region}_pct_agreement_0.05'] = \
            100 * agreement_df[f'{region}_agreement_0.05'].mean()
        agreement_summary[f'{region}_mean_std'] = \
            agreement_df[f'{region}_std'].mean()
        agreement_summary[f'{region}_mean_range'] = \
            agreement_df[f'{region}_range'].mean()
    
    return agreement_df, agreement_summary

def main():
    print("="*70)
    print("Test Set Evaluation - Comprehensive Analysis")
    print("="*70)
    
    all_results = []
    
    # Evaluate ensemble
    if ENSEMBLE_PRED_DIR.exists():
        ensemble_df = evaluate_all_cases(ENSEMBLE_PRED_DIR, 'Ensemble')
        all_results.append(ensemble_df)
    else:
        print(f"\nWarning: Ensemble predictions not found at {ENSEMBLE_PRED_DIR}")
        print("Please run ensemble prediction first.")
    
    # Evaluate individual folds
    for fold, pred_dir in FOLD_PRED_DIRS.items():
        if pred_dir.exists():
            fold_df = evaluate_all_cases(pred_dir, f'Fold_{fold}')
            all_results.append(fold_df)
        else:
            print(f"\nWarning: Fold {fold} predictions not found at {pred_dir}")
    
    if not all_results:
        print("\nERROR: No predictions found. Please run predictions first.")
        return
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Save per-case results
    results_csv = OUTPUT_DIR / "per_case_dice_scores.csv"
    all_results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Saved per-case results to: {results_csv}")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_stats = []
    
    for model in all_results_df['model'].unique():
        model_df = all_results_df[all_results_df['model'] == model]
        stats_dict = compute_statistics(model_df, model)
        summary_stats.append(stats_dict)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary statistics
    summary_csv = OUTPUT_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary statistics to: {summary_csv}")
    
    # Print summary table
    print("\n" + "="*70)
    print("Summary Statistics (Mean ± SD)")
    print("="*70)
    print(f"{'Model':<15} {'ED':<12} {'ET':<12} {'NCR':<12} {'TC':<12} {'WT':<12}")
    print("-"*70)
    
    for _, row in summary_df.iterrows():
        model = row['model']
        ed = f"{row['ED_mean']:.3f}±{row['ED_std']:.3f}"
        et = f"{row['ET_mean']:.3f}±{row['ET_std']:.3f}"
        ncr = f"{row['NCR_mean']:.3f}±{row['NCR_std']:.3f}"
        tc = f"{row['TC_mean']:.3f}±{row['TC_std']:.3f}"
        wt = f"{row['WT_mean']:.3f}±{row['WT_std']:.3f}"
        print(f"{model:<15} {ed:<12} {et:<12} {ncr:<12} {tc:<12} {wt:<12}")
    
    # Inter-fold agreement analysis
    if len([m for m in all_results_df['model'].unique() if m.startswith('Fold_')]) == 5:
        agreement_df, agreement_summary = compute_inter_fold_agreement(all_results_df)
        
        # Save agreement results
        agreement_csv = OUTPUT_DIR / "inter_fold_agreement.csv"
        agreement_df.to_csv(agreement_csv, index=False)
        print(f"\n✓ Saved inter-fold agreement to: {agreement_csv}")
        
        # Save agreement summary
        with open(OUTPUT_DIR / "agreement_summary.json", 'w') as f:
            json.dump(agreement_summary, f, indent=4)
        
        print("\n" + "="*70)
        print("Inter-Fold Agreement (% cases within 0.05 Dice)")
        print("="*70)
        for region in REGIONS.keys():
            pct = agreement_summary[f'{region}_pct_agreement_0.05']
            print(f"{region:<10}: {pct:>6.1f}%")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - per_case_dice_scores.csv")
    print("  - summary_statistics.csv")
    print("  - inter_fold_agreement.csv")
    print("  - agreement_summary.json")

if __name__ == "__main__":
    main()
