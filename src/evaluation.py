#!/usr/bin/env python3
"""
Evaluation utilities for SAM2 project
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from src.utils import dice_coefficient, iou_coefficient, load_image, load_mask, extract_class_masks, visualize_results


class SegmentationEvaluator:
    """Evaluation class for segmentation methods"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize evaluator
        
        Args:
            dataset_path: Path to CamVid dataset
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "val"
        self.labels_dir = self.dataset_path / "val_labels"
        
        # CamVid class mapping (based on RGB-to-class conversion)
        self.class_mapping = {
            'person': [7, 16],  # Child, Pedestrian
            'vehicle': [2, 5, 13, 22, 25, 27]  # Bicyclist, Car, MotorcycleScooter, SUVPickupTruck, Train, Truck_Bus
        }
        
        # Load class information if available
        self._load_class_info()
    
    def _load_class_info(self):
        """Load class information from dataset"""
        class_file = self.dataset_path / "class_dict.csv"
        if class_file.exists():
            try:
                import pandas as pd
                classes_df = pd.read_csv(class_file)
                print("Available classes:")
                print(classes_df)
                
                # Update class mapping based on actual dataset
                person_classes = []
                vehicle_classes = []
                
                for idx, row in classes_df.iterrows():
                    class_name = row['name'].lower()
                    class_id = row['id'] if 'id' in row else idx
                    
                    if any(keyword in class_name for keyword in ['person', 'pedestrian', 'human']):
                        person_classes.append(class_id)
                    elif any(keyword in class_name for keyword in ['car', 'vehicle', 'truck', 'bus', 'bike', 'motorcycle']):
                        vehicle_classes.append(class_id)
                
                if person_classes:
                    self.class_mapping['person'] = person_classes
                if vehicle_classes:
                    self.class_mapping['vehicle'] = vehicle_classes
                    
                print(f"Updated class mapping: {self.class_mapping}")
                
            except ImportError:
                print("pandas not available, using default class mapping")
            except Exception as e:
                print(f"Failed to load class info: {e}")
    
    def evaluate_method(self, 
                       segmentation_function,
                       method_name: str,
                       target_classes: List[str] = ['person', 'vehicle'],
                       max_images: int = None,
                       save_visualizations: bool = False,
                       viz_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate a segmentation method
        
        Args:
            segmentation_function: Function that takes image and returns dict of masks
            method_name: Name of the method being evaluated
            target_classes: Classes to evaluate
            max_images: Maximum number of images to evaluate (for testing)
            save_visualizations: Whether to save visualization images
            viz_dir: Directory to save visualizations
            
        Returns:
            Evaluation results dictionary
        """
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset not found at {self.dataset_path}")
        
        # Get image files
        image_files = list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg"))
        if max_images:
            image_files = image_files[:max_images]
        
        # Initialize results storage
        results = {class_name: {'dice': [], 'iou': []} for class_name in target_classes}
        results['overall'] = {'dice': [], 'iou': []}
        detailed_results = []
        
        # Setup visualization directory
        if save_visualizations and viz_dir:
            viz_path = Path(viz_dir)
            viz_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating {method_name} on {len(image_files)} images...")
        
        for img_idx, img_file in enumerate(tqdm(image_files)):
            try:
                # Load image
                image = load_image(img_file)
                
                # Load ground truth
                label_file = self.labels_dir / f"{img_file.stem}_L.png"
                if not label_file.exists():
                    # Try different extensions
                    label_file = self.labels_dir / f"{img_file.stem}_L.jpg"
                    if not label_file.exists():
                        print(f"Ground truth not found for {img_file.name}")
                        continue
                
                gt_label = load_mask(label_file)
                
                # Get predictions
                pred_masks = segmentation_function(image)
                
                # Evaluate each class
                image_results = {'filename': img_file.name}
                class_dice_scores = []
                class_iou_scores = []
                
                for class_name in target_classes:
                    if class_name in self.class_mapping and class_name in pred_masks:
                        # Extract ground truth mask for this class
                        gt_mask = extract_class_masks(gt_label, self.class_mapping[class_name])
                        pred_mask = pred_masks[class_name]
                        
                        # Calculate Dice and IoU scores
                        dice_score = dice_coefficient(pred_mask, gt_mask)
                        iou_score = iou_coefficient(pred_mask, gt_mask)
                        
                        results[class_name]['dice'].append(dice_score)
                        results[class_name]['iou'].append(iou_score)
                        class_dice_scores.append(dice_score)
                        class_iou_scores.append(iou_score)
                        
                        image_results[f'{class_name}_dice'] = float(dice_score)
                        image_results[f'{class_name}_iou'] = float(iou_score)
                        
                        # Save visualization if requested
                        if save_visualizations and viz_dir and img_idx < 10:  # Save first 10 for inspection
                            self._save_visualization(
                                image, gt_mask, pred_mask, 
                                viz_path / f"{img_file.stem}_{class_name}_{method_name}.png",
                                f"{method_name} - {class_name} - Dice: {dice_score:.3f}, IoU: {iou_score:.3f}"
                            )
                
                # Overall scores for this image
                if class_dice_scores:
                    overall_dice = np.mean(class_dice_scores)
                    overall_iou = np.mean(class_iou_scores)
                    results['overall']['dice'].append(overall_dice)
                    results['overall']['iou'].append(overall_iou)
                    image_results['overall_dice'] = float(overall_dice)
                    image_results['overall_iou'] = float(overall_iou)
                
                detailed_results.append(image_results)
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results, target_classes)
        
        # Add method info
        summary['method_name'] = method_name
        summary['total_images'] = len(image_files)
        summary['processed_images'] = len([r for r in detailed_results if 'overall_dice' in r])
        summary['detailed_results'] = detailed_results
        
        return summary
    
    def _calculate_summary_stats(self, results: Dict[str, Dict[str, List[float]]], target_classes: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics from results"""
        summary = {}
        
        for class_name in target_classes + ['overall']:
            if results[class_name]['dice']:
                dice_scores = np.array(results[class_name]['dice'])
                iou_scores = np.array(results[class_name]['iou'])
                
                summary[class_name] = {
                    'mean_dice': round(float(np.mean(dice_scores)), 2),
                    'std_dice': round(float(np.std(dice_scores)), 2),
                    'median_dice': round(float(np.median(dice_scores)), 2),
                    'min_dice': round(float(np.min(dice_scores)), 2),
                    'max_dice': round(float(np.max(dice_scores)), 2),
                    'mean_iou': round(float(np.mean(iou_scores)), 2),
                    'std_iou': round(float(np.std(iou_scores)), 2),
                    'median_iou': round(float(np.median(iou_scores)), 2),
                    'min_iou': round(float(np.min(iou_scores)), 2),
                    'max_iou': round(float(np.max(iou_scores)), 2),
                    'num_images': int(len(dice_scores))
                }
            else:
                summary[class_name] = {
                    'mean_dice': 0.0,
                    'std_dice': 0.0,
                    'median_dice': 0.0,
                    'min_dice': 0.0,
                    'max_dice': 0.0,
                    'mean_iou': 0.0,
                    'std_iou': 0.0,
                    'median_iou': 0.0,
                    'min_iou': 0.0,
                    'max_iou': 0.0,
                    'num_images': 0
                }
        
        return summary
    
    def _save_visualization(self, image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, 
                          save_path: Path, title: str):
        """Save visualization comparing GT and prediction"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        # Overlay comparison
        overlay = image.copy()
        overlay[gt_mask > 0] = [0, 255, 0]  # Green for GT
        overlay[pred_mask > 0] = [255, 0, 0]  # Red for prediction
        overlay[(gt_mask > 0) & (pred_mask > 0)] = [255, 255, 0]  # Yellow for overlap
        
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (GT=Green, Pred=Red, Overlap=Yellow)")
        axes[3].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def compare_methods(self, results_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Compare multiple methods and print results
        
        Args:
            results_dict: Dictionary mapping method names to their results
        """
        print("\n" + "="*80)
        print("METHOD COMPARISON")
        print("="*80)
        
        # Print header
        print(f"{'Method':<20} {'Person Dice':<12} {'Vehicle Dice':<12} {'Overall Dice':<12} {'Images':<8}")
        print("-" * 80)
        
        # Print results for each method
        for method_name, results in results_dict.items():
            person_dice = results.get('person', {}).get('mean_dice', 0.0)
            vehicle_dice = results.get('vehicle', {}).get('mean_dice', 0.0)
            overall_dice = results.get('overall', {}).get('mean_dice', 0.0)
            num_images = results.get('processed_images', 0)
            
            print(f"{method_name:<20} {person_dice:<12.4f} {vehicle_dice:<12.4f} {overall_dice:<12.4f} {num_images:<8}")
        
        print("-" * 80)
        
        # Find best method
        best_method = max(results_dict.keys(), 
                         key=lambda x: results_dict[x].get('overall', {}).get('mean_dice', 0.0))
        best_score = results_dict[best_method].get('overall', {}).get('mean_dice', 0.0)
        
        print(f"\nBest Method: {best_method} (Overall Dice: {best_score:.4f})")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Test the evaluation system"""
    dataset_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/camvid"
    
    if not Path(dataset_path).exists():
        print("Dataset not found. Please run data preparation first.")
        return
    
    evaluator = SegmentationEvaluator(dataset_path)
    
    # Test with a dummy segmentation function
    def dummy_segmentation(image):
        h, w = image.shape[:2]
        return {
            'person': np.random.rand(h, w) > 0.8,
            'vehicle': np.random.rand(h, w) > 0.9
        }
    
    results = evaluator.evaluate_method(
        dummy_segmentation, 
        "dummy_test", 
        max_images=5,
        save_visualizations=True,
        viz_dir="/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results/test_viz"
    )
    
    print("Test results:")
    for class_name, metrics in results.items():
        if isinstance(metrics, dict) and 'mean_dice' in metrics:
            print(f"{class_name}: {metrics['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
