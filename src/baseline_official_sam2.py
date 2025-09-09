#!/usr/bin/env python3
"""
Official SAM2 Automatic Mask Generator Baseline
Following the official SAM2 repository methodology

This is the proper baseline using SAM2AutomaticMaskGenerator from the official repo
that automatically generates masks without manual prompts.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add official SAM2 repo to path
sys.path.insert(0, '/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/official_sam2_repo')
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel')

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("Official SAM2 imported successfully!")
except ImportError as e:
    print(f"Failed to import official SAM2: {e}")

# Import utility functions
from src.utils import load_image, load_mask, dice_coefficient, iou_coefficient, extract_class_masks


def show_anns(anns, borders=True):
    """
    Official SAM2 visualization function from the repository
    Displays all masks overlayed on an image with random colors
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


class OfficialSAM2AutomaticBaseline:
    """
    Official SAM2 Automatic Baseline using SAM2AutomaticMaskGenerator
    
    This follows the official repository methodology and automatically generates
    masks without requiring any manual prompts, clicks, or boxes.
    """
    
    def __init__(self, model_cfg: str = "sam2_hiera_l.yaml"):
        """
        Initialize official SAM2 automatic baseline
        
        Args:
            model_cfg: SAM2 model configuration
        """
        self.model_cfg = model_cfg
        
        # Load SAM2 model using official methodology
        self.sam2_model = self._load_sam2_model()
        
        # Initialize automatic mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=32,  # Standard setting from official repo
            pred_iou_thresh=0.8,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Remove small masks
            use_m2m=True,  # Use mask-to-mask refinement
        )
        
        # CamVid class mapping for target classes
        self.class_mapping = {
            'person': [7, 16],  # Child, Pedestrian  
            'vehicle': [2, 5, 13, 22, 25, 27]  # Bicyclist, Car, MotorcycleScooter, SUVPickupTruck, Train, Truck_Bus
        }
    
    def _load_sam2_model(self):
        """Load SAM2 model using official repository approach"""
        try:
            # Get checkpoint and config paths
            checkpoint_path = self._get_checkpoint_path()
            config_path = self._get_config_path()
            
            # Build SAM2 model following official methodology
            sam2_model = build_sam2(config_path, checkpoint_path, device="cuda")
            print("Official SAM2 model loaded successfully!")
            return sam2_model
        except Exception as e:
            print(f"Failed to load SAM2 model: {e}")
            return None
    
    def _get_checkpoint_path(self) -> str:
        """Get SAM2 checkpoint path"""
        # Check official repo checkpoints first
        official_checkpoint_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/official_sam2_repo/checkpoints")
        if official_checkpoint_dir.exists():
            checkpoint_files = list(official_checkpoint_dir.glob("*.pt"))
            if checkpoint_files:
                return str(checkpoint_files[0])
        
        # Fallback to our checkpoints
        checkpoint_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/checkpoints")
        checkpoint_file = checkpoint_dir / "sam2_hiera_large.pt"
        
        if checkpoint_file.exists():
            return str(checkpoint_file)
        else:
            # Try other names
            possible_paths = [
                checkpoint_dir / "sam2_hiera_l.pt",
                checkpoint_dir / "sam2.pt"
            ]
            for path in possible_paths:
                if path.exists():
                    return str(path)
        
        raise FileNotFoundError(f"SAM2 checkpoint not found")
    
    def _get_config_path(self) -> str:
        """Get SAM2 config path"""
        # The config file should just be the yaml filename, not full path
        # SAM2 will find it in its config directory
        return self.model_cfg
    
    def segment_automatically(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Automatically segment image using official SAM2AutomaticMaskGenerator
        
        This generates all masks automatically without any manual intervention,
        then filters them for person and vehicle classes.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Dictionary with 'person' and 'vehicle' binary masks
        """
        if self.mask_generator is None:
            print("SAM2 automatic mask generator not initialized")
            return {
                'person': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
                'vehicle': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            }
        
        try:
            # Generate all masks automatically using official SAM2
            masks = self.mask_generator.generate(image)
            
            # Convert to binary masks for person and vehicle detection
            person_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            vehicle_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Since SAM2 doesn't know about specific classes, we use a heuristic approach
            # to assign generated masks to person/vehicle categories based on properties
            for mask_data in masks:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, w, h]
                area = mask_data['area']
                
                # Heuristic classification based on mask properties
                if self._is_likely_person(bbox, area, image.shape):
                    person_mask = np.logical_or(person_mask, mask).astype(np.uint8)
                elif self._is_likely_vehicle(bbox, area, image.shape):
                    vehicle_mask = np.logical_or(vehicle_mask, mask).astype(np.uint8)
            
            return {
                'person': person_mask,
                'vehicle': vehicle_mask
            }
            
        except Exception as e:
            print(f"Automatic mask generation failed: {e}")
            return {
                'person': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
                'vehicle': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            }
    
    def _is_likely_person(self, bbox: List[float], area: int, image_shape: Tuple[int, int, int]) -> bool:
        """
        Heuristic to determine if a mask is likely a person
        
        Args:
            bbox: Bounding box [x, y, w, h]
            area: Mask area
            image_shape: Image shape (H, W, C)
            
        Returns:
            True if likely a person
        """
        x, y, w, h = bbox
        height, width = image_shape[:2]
        
        # Person heuristics for road scenes:
        # - Typical aspect ratio (taller than wide)
        # - Medium size (not too large, not too small)
        # - Position in upper-middle regions
        aspect_ratio = h / max(w, 1)
        relative_area = area / (height * width)
        center_y = y + h/2
        relative_y = center_y / height
        
        return (
            1.5 <= aspect_ratio <= 4.0 and  # Tall objects
            0.001 <= relative_area <= 0.1 and  # Medium size
            0.2 <= relative_y <= 0.8  # Upper-middle regions
        )
    
    def _is_likely_vehicle(self, bbox: List[float], area: int, image_shape: Tuple[int, int, int]) -> bool:
        """
        Heuristic to determine if a mask is likely a vehicle
        
        Args:
            bbox: Bounding box [x, y, w, h]
            area: Mask area
            image_shape: Image shape (H, W, C)
            
        Returns:
            True if likely a vehicle
        """
        x, y, w, h = bbox
        height, width = image_shape[:2]
        
        # Vehicle heuristics for road scenes:
        # - Wider than tall or square-ish
        # - Larger size
        # - Position in lower regions (roads)
        aspect_ratio = h / max(w, 1)
        relative_area = area / (height * width)
        center_y = y + h/2
        relative_y = center_y / height
        
        return (
            0.3 <= aspect_ratio <= 1.5 and  # Wide or square objects
            0.01 <= relative_area <= 0.5 and  # Large size
            0.4 <= relative_y <= 1.0  # Lower regions (roads)
        )
    
    def evaluate_on_dataset(self,
                           dataset_path: str,
                           gt_path: str,
                           num_images: int = 100,
                           save_results: bool = True,
                           save_visualizations: bool = True,
                           viz_dir: str = "results/baseline/visualizations") -> Dict[str, Any]:
        """
        Evaluate official SAM2 automatic baseline on CamVid dataset
        
        Args:
            dataset_path: Path to CamVid images
            gt_path: Path to ground truth masks
            num_images: Number of images to evaluate
            save_results: Whether to save detailed results
            save_visualizations: Whether to save visualization images
            viz_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Starting Official SAM2 Automatic Baseline Evaluation...")
        print("Using SAM2AutomaticMaskGenerator from official repository")
        
        image_dir = Path(dataset_path)
        gt_dir = Path(gt_path)
        
        # Setup visualization directory
        if save_visualizations:
            viz_path = Path(viz_dir)
            viz_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving visualizations to: {viz_path}")
        
        # Get list of validation images  
        image_files = sorted([f for f in image_dir.glob("*.png") if f.is_file()])[:num_images]
        
        results = {
            'person': {'dice_scores': [], 'iou_scores': []},
            'vehicle': {'dice_scores': [], 'iou_scores': []},
            'predictions': [],
            'config': {
                'method': 'official_sam2_automatic',
                'model_cfg': self.model_cfg,
                'uses_automatic_mask_generator': True
            }
        }
        
        total_time = 0
        
        for i, image_file in enumerate(tqdm(image_files, desc="Evaluating Official SAM2 Automatic")):
            try:
                # Load image and ground truth
                image = load_image(str(image_file))
                
                # GT file has _L suffix
                gt_filename = image_file.stem + "_L.png"
                gt_file = gt_dir / gt_filename
                
                if not gt_file.exists():
                    continue
                    
                gt_mask = load_mask(str(gt_file))
                
                # Extract class-specific masks using class mapping
                person_gt = extract_class_masks(gt_mask, self.class_mapping['person'])
                vehicle_gt = extract_class_masks(gt_mask, self.class_mapping['vehicle'])
                
                import time
                start_time = time.time()
                
                # Automatically generate masks using official SAM2
                predictions = self.segment_automatically(image)
                person_pred = predictions['person']
                vehicle_pred = predictions['vehicle']
                
                # Save visualization if requested (for first 10 images)
                if save_visualizations and i < 10:
                    self._save_official_visualization(image, image_file.stem, viz_path, i)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
                # Initialize metrics
                person_dice = None
                person_iou = None
                vehicle_dice = None
                vehicle_iou = None
                
                # Calculate metrics for person
                if np.sum(person_gt) > 0:  # Only if person exists in GT
                    person_dice = dice_coefficient(person_pred, person_gt)
                    person_iou = iou_coefficient(person_pred, person_gt)
                    results['person']['dice_scores'].append(person_dice)
                    results['person']['iou_scores'].append(person_iou)
                
                # Calculate metrics for vehicle
                if np.sum(vehicle_gt) > 0:  # Only if vehicle exists in GT
                    vehicle_dice = dice_coefficient(vehicle_pred, vehicle_gt)
                    vehicle_iou = iou_coefficient(vehicle_pred, vehicle_gt)
                    results['vehicle']['dice_scores'].append(vehicle_dice)
                    results['vehicle']['iou_scores'].append(vehicle_iou)
                
                # Store prediction details
                results['predictions'].append({
                    'image': image_file.name,
                    'person_dice': float(person_dice) if person_dice is not None else None,
                    'person_iou': float(person_iou) if person_iou is not None else None,
                    'vehicle_dice': float(vehicle_dice) if vehicle_dice is not None else None,
                    'vehicle_iou': float(vehicle_iou) if vehicle_iou is not None else None,
                })
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        # Calculate final metrics
        person_dice_mean = np.mean(results['person']['dice_scores']) if results['person']['dice_scores'] else 0.0
        person_iou_mean = np.mean(results['person']['iou_scores']) if results['person']['iou_scores'] else 0.0
        vehicle_dice_mean = np.mean(results['vehicle']['dice_scores']) if results['vehicle']['dice_scores'] else 0.0
        vehicle_iou_mean = np.mean(results['vehicle']['iou_scores']) if results['vehicle']['iou_scores'] else 0.0
        
        # Overall metrics
        total_dice_scores = results['person']['dice_scores'] + results['vehicle']['dice_scores']
        total_iou_scores = results['person']['iou_scores'] + results['vehicle']['iou_scores']
        overall_dice = np.mean(total_dice_scores) if total_dice_scores else 0.0
        overall_iou = np.mean(total_iou_scores) if total_iou_scores else 0.0
        
        # Performance metrics
        avg_time_per_image = total_time / len(image_files) if len(image_files) > 0 else 0.0
        
        final_results = {
            'person_dice': round(float(person_dice_mean), 2),
            'person_iou': round(float(person_iou_mean), 2), 
            'vehicle_dice': round(float(vehicle_dice_mean), 2),
            'vehicle_iou': round(float(vehicle_iou_mean), 2),
            'overall_dice': round(float(overall_dice), 2),
            'overall_iou': round(float(overall_iou), 2),
            'total_time': round(float(total_time), 2),
            'avg_time_per_image': round(float(avg_time_per_image), 2),
            'num_images': len(image_files),
            'num_person_samples': len(results['person']['dice_scores']),
            'num_vehicle_samples': len(results['vehicle']['dice_scores']),
            'config': results['config']
        }
        
        # Save results
        if save_results:
            results_path = Path("results/official_sam2_automatic_baseline_results.json")
            results_path.parent.mkdir(exist_ok=True)
            
            # Save detailed results
            detailed_results = {
                'final_metrics': final_results,
                'detailed_predictions': results['predictions'],
                'config': results['config']
            }
            
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"Results saved to: {results_path}")
        
        return final_results
    
    def _save_official_visualization(self, image: np.ndarray, image_name: str, viz_dir: Path, img_idx: int):
        """
        Save visualization using official SAM2 style from the repository
        Shows all automatically generated masks overlayed on the image
        """
        try:
            # Generate masks for visualization
            if self.mask_generator is None:
                return
                
            masks = self.mask_generator.generate(image)
            
            # Create visualization using official show_anns function
            plt.figure(figsize=(20, 20))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plt.title(f'Official SAM2 Automatic Masks - {image_name}', fontsize=16)
            
            # Save the visualization
            save_path = viz_dir / f"{img_idx:03d}_{image_name}_automatic_masks.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Saved visualization: {save_path}")
            
        except Exception as e:
            print(f"Error saving visualization for {image_name}: {e}")


if __name__ == "__main__":
    # Test the official SAM2 automatic baseline
    baseline = OfficialSAM2AutomaticBaseline()
    
    # Test on a few images
    dataset_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid/val"
    gt_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid/val_labels"
    
    results = baseline.evaluate_on_dataset(dataset_path, gt_path, num_images=10)
    print("Official SAM2 automatic baseline results:")
    print(f"Person Dice: {results['person_dice']:.4f}")
    print(f"Vehicle Dice: {results['vehicle_dice']:.4f}")
    print(f"Overall Dice: {results['overall_dice']:.4f}")
