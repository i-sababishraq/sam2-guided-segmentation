#!/usr/bin/env python3
"""
Utility functions for the SAM2 project
"""
import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient between predicted and true masks
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    true_mask = (true_mask > 0.5).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(pred_mask * true_mask)
    total = np.sum(pred_mask) + np.sum(true_mask)
    
    if total == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    return (2.0 * intersection) / total

def iou_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) coefficient between predicted and true masks
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        IoU coefficient (0-1, higher is better)
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    true_mask = (true_mask > 0.5).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    
    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    return intersection / union

def load_image(image_path: Path) -> np.ndarray:
    """Load and preprocess image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_mask(mask_path: Path) -> np.ndarray:
    """Load ground truth mask and convert from RGB to class indices for CamVid"""
    # Load as RGB image
    mask_rgb = cv2.imread(str(mask_path))
    if mask_rgb is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    
    # CamVid RGB to class mapping
    rgb_to_class = {
        (64, 128, 64): 0,    # Animal
        (192, 0, 128): 1,    # Archway
        (0, 128, 192): 2,    # Bicyclist
        (0, 128, 64): 3,     # Bridge
        (128, 0, 0): 4,      # Building
        (64, 0, 128): 5,     # Car
        (64, 0, 192): 6,     # CartLuggagePram
        (192, 128, 64): 7,   # Child
        (192, 192, 128): 8,  # Column_Pole
        (64, 64, 128): 9,    # Fence
        (128, 0, 192): 10,   # LaneMkgsDriv
        (192, 0, 64): 11,    # LaneMkgsNonDriv
        (128, 128, 64): 12,  # Misc_Text
        (192, 0, 192): 13,   # MotorcycleScooter
        (128, 64, 64): 14,   # OtherMoving
        (64, 192, 128): 15,  # ParkingBlock
        (64, 64, 0): 16,     # Pedestrian
        (128, 64, 128): 17,  # Road
        (128, 128, 192): 18, # RoadShoulder
        (0, 0, 192): 19,     # Sidewalk
        (192, 128, 128): 20, # SignSymbol
        (128, 128, 128): 21, # Sky
        (64, 128, 192): 22,  # SUVPickupTruck
        (0, 0, 64): 23,      # TrafficCone
        (0, 64, 64): 24,     # TrafficLight
        (192, 64, 128): 25,  # Train
        (128, 128, 0): 26,   # Tree
        (192, 128, 192): 27, # Truck_Bus
        (64, 0, 64): 28,     # Tunnel
        (192, 192, 0): 29,   # VegetationMisc
        (0, 0, 0): 30,       # Void
        (64, 192, 0): 31     # Wall
    }
    
    # Convert RGB to class indices
    height, width = mask_rgb.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)
    
    for (r, g, b), class_id in rgb_to_class.items():
        mask = (mask_rgb[:,:,0] == r) & (mask_rgb[:,:,1] == g) & (mask_rgb[:,:,2] == b)
        class_mask[mask] = class_id
    
    return class_mask

def visualize_results(image: np.ndarray, 
                     gt_mask: np.ndarray, 
                     pred_mask: np.ndarray, 
                     title: str = "Results") -> None:
    """Visualize image, ground truth, and prediction"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def extract_class_masks(label_mask: np.ndarray, 
                       target_classes: List[int]) -> np.ndarray:
    """
    Extract binary masks for specific classes from multi-class label
    
    Args:
        label_mask: Multi-class segmentation mask
        target_classes: List of class IDs to extract
        
    Returns:
        Binary mask with 1 for target classes, 0 otherwise
    """
    binary_mask = np.zeros_like(label_mask, dtype=np.uint8)
    for class_id in target_classes:
        binary_mask[label_mask == class_id] = 1
    return binary_mask

def get_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from binary mask
    
    Args:
        mask: Binary mask
        
    Returns:
        List of bounding boxes in format (x1, y1, x2, y2)
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    boxes = []
    for label_id in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            boxes.append((x, y, x + w, y + h))
    
    return boxes

def get_center_points(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract center points from binary mask
    
    Args:
        mask: Binary mask
        
    Returns:
        List of center points in format (x, y)
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    points = []
    for label_id in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label_id).astype(np.uint8)
        
        # Calculate centroid
        M = cv2.moments(component_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    
    return points

def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to file"""
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def load_sam2_model(device: str = 'cuda'):
    """Load SAM2 model and predictor"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Updated config and checkpoint paths
        sam2_checkpoint = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"  # Fixed config name
        
        # Build model
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return sam2_model, predictor
        
    except Exception as e:
        print(f"Failed to load SAM2 model: {e}")
        return None, None
