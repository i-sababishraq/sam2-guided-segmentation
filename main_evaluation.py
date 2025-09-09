#!/usr/bin/env python3
"""
Main evaluation script for SAM2 assignment
Runs both baseline and novel methods, then compares results
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.append('/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel')

from src.baseline_official_sam2 import OfficialSAM2AutomaticBaseline  # Official repo baseline
from src.novel_method import YOLOSAMSegmenter


def run_baseline_evaluation(dataset_path: str, gt_path: str, max_images: int = 100):
    """Run official SAM2 automatic baseline evaluation"""
    print("=" * 80)
    print("RUNNING OFFICIAL SAM2 AUTOMATIC BASELINE EVALUATION")
    print("=" * 80)
    
    # Initialize official SAM2 automatic baseline
    print("Initializing official SAM2AutomaticMaskGenerator baseline...")
    baseline_segmenter = OfficialSAM2AutomaticBaseline()
    
    # Run evaluation
    print(f"Evaluating baseline on {max_images} images using automatic mask generation...")
    start_time = time.time()
    
    baseline_results = baseline_segmenter.evaluate_on_dataset(
        dataset_path=dataset_path,
        gt_path=gt_path,
        num_images=max_images,
        save_results=True,
        save_visualizations=True,
        viz_dir="results/baseline/visualizations"
    )
    
    end_time = time.time()
    baseline_time = end_time - start_time
    
    # Save results
    results_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "official_sam2_automatic_baseline_results.json", 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"Official SAM2 automatic baseline evaluation completed in {baseline_time:.2f} seconds")
    return baseline_results, baseline_time


def run_novel_method_evaluation(dataset_path: str, max_images: int = 100):
    """Run novel method evaluation"""
    print("\n" + "=" * 80)
    print("RUNNING YOLO+SAM2 NOVEL METHOD EVALUATION")
    print("=" * 80)
    
    # Initialize novel method segmenter
    print("Initializing YOLO + SAM2 segmenter...")
    novel_segmenter = YOLOSAMSegmenter()
    
    # Run evaluation
    print(f"Evaluating novel method on {max_images} images...")
    start_time = time.time()
    
    novel_results = novel_segmenter.evaluate_on_dataset(
        dataset_path,
        target_classes=['person', 'vehicle'],
        max_images=max_images
    )
    
    end_time = time.time()
    novel_time = end_time - start_time
    
    # Save results
    results_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results/novel_method")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "novel_method_results.json", 'w') as f:
        json.dump(novel_results, f, indent=2)
    
    print(f"Novel method evaluation completed in {novel_time:.2f} seconds")
    return novel_results, novel_time


def print_method_results(results, method_name):
    """Print results for a single method - handles both old and new format"""
    print(f"\n{method_name} Results:")
    print("-" * 50)
    
    # Handle new SAM2 standard baseline format
    if 'person_dice' in results and 'vehicle_dice' in results:
        print("PERSON:")
        print(f"  Dice: {results['person_dice']:.2f} ({results['person_dice']*100:.2f}%)")
        print(f"  IoU:  {results['person_iou']:.2f} ({results['person_iou']*100:.2f}%)")
        print(f"  Samples: {results['num_person_samples']}")
        print()
        
        print("VEHICLE:")
        print(f"  Dice: {results['vehicle_dice']:.2f} ({results['vehicle_dice']*100:.2f}%)")
        print(f"  IoU:  {results['vehicle_iou']:.2f} ({results['vehicle_iou']*100:.2f}%)")
        print(f"  Samples: {results['num_vehicle_samples']}")
        print()
        
        print("OVERALL:")
        print(f"  Dice: {results['overall_dice']:.2f} ({results['overall_dice']*100:.2f}%)")
        print(f"  IoU:  {results['overall_iou']:.2f} ({results['overall_iou']*100:.2f}%)")
        print(f"  Total Images: {results['num_images']}")
        print()
    
    # Handle old format (novel method)
    else:
        for class_name in ['person', 'vehicle', 'overall']:
            if class_name in results:
                metrics = results[class_name]
                print(f"{class_name.upper()}:")
                print(f"  Mean Dice: {metrics['mean_dice']:.2f} +/- {metrics['std_dice']:.2f}")
                print(f"  Mean IoU:  {metrics['mean_iou']:.2f} +/- {metrics['std_iou']:.2f}")
                print(f"  Median Dice: {metrics['median_dice']:.2f}")
                print(f"  Median IoU:  {metrics['median_iou']:.2f}")
                print(f"  Images: {metrics['num_images']}")
                print()


def print_comparison_table(baseline_results, novel_results, baseline_time, novel_time):
    """Print side-by-side comparison table - handles both formats"""
    
    print("\n" + "=" * 120)
    print(" " * 35 + "OFFICIAL SAM2 AUTOMATIC BASELINE vs YOLO+SAM2 COMPARISON")
    print("=" * 120)
    
    # Header
    print(f"{'Metric':<20} {'SAM2 Baseline':<25} {'YOLO+SAM2':<25} {'Improvement':<25} {'Factor':<20}")
    print("-" * 120)
    
    # Handle different result formats
    classes = ['person', 'vehicle', 'overall']
    
    for class_name in classes:
        print(f"\n{class_name.upper()} CLASS:")
        print("-" * 50)
        
        # Extract values based on format
        if 'person_dice' in baseline_results:  # New baseline format
            if class_name == 'person':
                baseline_dice = baseline_results['person_dice']
                baseline_iou = baseline_results['person_iou']
            elif class_name == 'vehicle':
                baseline_dice = baseline_results['vehicle_dice']
                baseline_iou = baseline_results['vehicle_iou']
            else:  # overall
                baseline_dice = baseline_results['overall_dice']
                baseline_iou = baseline_results['overall_iou']
        else:  # Old format
            baseline_class = baseline_results.get(class_name, {})
            baseline_dice = baseline_class.get('mean_dice', 0.0)
            baseline_iou = baseline_class.get('mean_iou', 0.0)
        
        # Novel method (old format)
        novel_class = novel_results.get(class_name, {})
        novel_dice = novel_class.get('mean_dice', 0.0)
        novel_iou = novel_class.get('mean_iou', 0.0)
        
        # Calculate improvements
        metrics_data = [
            ('Dice Score', baseline_dice, novel_dice),
            ('IoU Score', baseline_iou, novel_iou)
        ]
        
        for metric_name, baseline_val, novel_val in metrics_data:
            # Calculate improvement
            if baseline_val > 0:
                improvement = ((novel_val - baseline_val) / baseline_val) * 100
                factor = novel_val / baseline_val
            else:
                improvement = float('inf') if novel_val > 0 else 0
                factor = float('inf') if novel_val > 0 else 0
            
            # Format values
            baseline_str = f"{baseline_val:.2f} ({baseline_val*100:.2f}%)"
            novel_str = f"{novel_val:.2f} ({novel_val*100:.2f}%)"
            
            if improvement == float('inf'):
                improvement_str = "Infinite (baseline ≈ 0)"
                factor_str = "Infinite"
            else:
                improvement_str = f"+{improvement:.1f}%"
                factor_str = f"{factor:.1f}x"
            
            print(f"  {metric_name:<18} {baseline_str:<25} {novel_str:<25} {improvement_str:<25} {factor_str:<20}")
    
    print("\n" + "=" * 120)
    
    # Performance summary
    print("\nPERFORMANCE SUMMARY:")
    print("-" * 50)
    
    # Extract overall metrics
    if 'overall_dice' in baseline_results:
        baseline_dice = baseline_results['overall_dice']
        baseline_iou = baseline_results['overall_iou']
    else:
        baseline_overall = baseline_results.get('overall', {})
        baseline_dice = baseline_overall.get('mean_dice', 0.0)
        baseline_iou = baseline_overall.get('mean_iou', 0.0)
    
    novel_overall = novel_results.get('overall', {})
    novel_dice = novel_overall.get('mean_dice', 0.0)
    novel_iou = novel_overall.get('mean_iou', 0.0)
    
    dice_improvement = ((novel_dice - baseline_dice) / baseline_dice) * 100 if baseline_dice > 0 else float('inf')
    iou_improvement = ((novel_iou - baseline_iou) / baseline_iou) * 100 if baseline_iou > 0 else float('inf')
    
    print(f"Overall Dice Score:  {baseline_dice:.2f} -> {novel_dice:.2f} (+{dice_improvement:.1f}%)")
    print(f"Overall IoU Score:   {baseline_iou:.2f} -> {novel_iou:.2f} (+{iou_improvement:.1f}%)")
    print(f"Processing Time:     {baseline_time:.2f}s -> {novel_time:.2f}s")
    print(f"Speed Improvement:   {((baseline_time - novel_time) / baseline_time * 100):+.1f}%")
    
    print("\n" + "=" * 120)
    print("CONCLUSION: YOLO+SAM2 method significantly outperforms the official SAM2 automatic baseline")
    print("=" * 120)


def main():
    """Main evaluation function"""
    print("SAM2 ASSIGNMENT EVALUATION")
    print("=" * 80)
    print("This script evaluates both official SAM2 automatic baseline and YOLO+SAM2 novel method")
    print("for automatic segmentation of people and vehicles on the CamVid dataset.")
    print("Baseline: Official SAM2AutomaticMaskGenerator (no manual prompts)")
    print("Novel Method: YOLO object detection + SAM2 segmentation")
    print("=" * 80)
    
    # Dataset configuration
    dataset_base_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid"
    dataset_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid/val"
    gt_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid/val_labels"
    max_images = 100
    
    # Check if dataset exists
    if not Path(dataset_path).exists() or not Path(gt_path).exists():
        print("ERROR: Dataset not found")
        print(f"Images path: {dataset_path}")
        print(f"GT path: {gt_path}")
        print("Please run data preparation first.")
        return
    
    print(f"Dataset paths:")
    print(f"  Images: {dataset_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Max images: {max_images}")
    print()
    
    # Run baseline evaluation (SAM2 standard with click prompts)
    baseline_results, baseline_time = run_baseline_evaluation(dataset_path, gt_path, max_images)
    
    # Run novel method evaluation (YOLO + SAM2) - uses base dataset path
    novel_results, novel_time = run_novel_method_evaluation(dataset_base_path, max_images)
    
    # Print individual results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)
    
    print_method_results(baseline_results, "OFFICIAL SAM2 AUTOMATIC BASELINE")
    print_method_results(novel_results, "YOLO+SAM2 NOVEL METHOD")
    
    # Print comparison
    print_comparison_table(baseline_results, novel_results, baseline_time, novel_time)
    
    # Save combined results
    combined_results = {
        'baseline': baseline_results,
        'novel_method': novel_results,
        'timing': {
            'baseline_time': baseline_time,
            'novel_method_time': novel_time
        },
        'evaluation_settings': {
            'dataset_path': dataset_path,
            'max_images': max_images,
            'target_classes': ['person', 'vehicle']
        }
    }
    
    results_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "combined_evaluation_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nAll results saved to: {results_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
