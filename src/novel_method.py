#!/usr/bin/env python3
"""
Novel Method: Object Detection + SAM2 Segmentation

This method uses YOLOv8 for object detection to automatically generate 
bounding box prompts for SAM2, eliminating the need for manual prompts.
"""
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import (
    load_sam2_model, load_image, load_mask, 
    dice_coefficient, iou_coefficient, extract_class_masks
)


class YOLOSAMSegmenter:
    """Novel method combining YOLO object detection with SAM2 segmentation"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize YOLO + SAM2 segmenter
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize model variables
        self.yolo_model = None
        self.sam2_model = None
        self.predictor = None
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano version for speed
            print("YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
        
        # Load SAM2 model
        print("Loading SAM2 model...")
        try:
            self.sam2_model, self.predictor = load_sam2_model(device=device)
            if self.predictor is not None:
                print("SAM2 model loaded successfully!")
            else:
                print("SAM2 predictor is None")
        except Exception as e:
            print(f"Failed to load SAM2 model: {e}")
        
        # YOLO class mapping to our target classes
        # YOLO COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, 1=bicycle
        self.yolo_to_target = {
            'person': [0],  # person
            'vehicle': [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        }
        
        # Confidence threshold for YOLO detections
        self.confidence_threshold = 0.25
        
    def detect_objects(self, image: np.ndarray, target_classes: List[str]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect objects using YOLO and return bounding boxes for target classes
        
        Args:
            image: Input image
            target_classes: List of class names to detect ('person', 'vehicle')
            
        Returns:
            Dictionary mapping class names to lists of bounding boxes (x1, y1, x2, y2)
        """
        # Run YOLO detection
        results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
        
        # Initialize output
        detections = {class_name: [] for class_name in target_classes}
        
        if len(results) == 0:
            return detections
            
        # Process detections
        result = results[0]  # Single image
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # Group by target classes
            for box, conf, cls in zip(boxes, confidences, classes):
                for target_class in target_classes:
                    if cls in self.yolo_to_target[target_class]:
                        x1, y1, x2, y2 = box.astype(int)
                        detections[target_class].append((x1, y1, x2, y2))
                        break
        
        return detections
    
    def segment_with_bbox_prompt(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Segment object using bounding box as prompt for SAM2
        
        Args:
            image: Input image  
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Binary segmentation mask
        """
        if self.predictor is None:
            print("SAM2 model not loaded")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Set image for SAM2
        self.predictor.set_image(image)
        
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        try:
            # Use bounding box as prompt
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],  # Add batch dimension
                multimask_output=False
            )
            
            # Return the mask
            mask = masks[0] if len(masks) > 0 else np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            return mask.astype(np.uint8)
            
        except Exception as e:
            print(f"SAM2 prediction failed: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def segment_image(self, image: np.ndarray, target_classes: List[str]) -> Dict[str, np.ndarray]:
        """
        Segment image using YOLO detection + SAM2 segmentation
        
        Args:
            image: Input image
            target_classes: List of class names to segment
            
        Returns:
            Dictionary mapping class names to binary masks
        """
        # Step 1: Detect objects with YOLO
        detections = self.detect_objects(image, target_classes)
        
        # Step 2: Segment each detection with SAM2
        results = {}
        
        for class_name in target_classes:
            # Combine all masks for this class
            class_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            for bbox in detections[class_name]:
                # Segment this detection
                mask = self.segment_with_bbox_prompt(image, bbox)
                
                # Combine with existing mask for this class
                class_mask = np.logical_or(class_mask, mask).astype(np.uint8)
            
            results[class_name] = class_mask
        
        return results
    
    def evaluate_on_dataset(self, 
                           dataset_path: str, 
                           target_classes: List[str] = ['person', 'vehicle'],
                           max_images: int = None) -> Dict[str, Any]:
        """
        Evaluate novel method on dataset
        
        Args:
            dataset_path: Path to dataset
            target_classes: Classes to segment
            max_images: Maximum number of images to evaluate (for testing)
            
        Returns:
            Evaluation results
        """
        # Use the evaluation framework
        from src.evaluation import SegmentationEvaluator
        
        evaluator = SegmentationEvaluator(dataset_path)
        
        # Define segmentation function for evaluator
        def segment_func(image):
            return self.segment_image(image, target_classes)
        
        # Run evaluation
        results = evaluator.evaluate_method(
            segmentation_function=segment_func,
            target_classes=target_classes,
            method_name="YOLO+SAM2",
            max_images=max_images,
            save_visualizations=True,
            viz_dir="results/novel_method/visualizations"
        )
        
        return results


def main():
    """Main function to run novel method evaluation"""
    
    # Initialize segmenter
    print("Initializing YOLO + SAM2 segmenter...")
    segmenter = YOLOSAMSegmenter()
    
    # Dataset path
    dataset_path = "/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/data/CamVid"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print("Dataset not found. Please run data preparation first.")
        return
    
    # Evaluate novel method
    print("Running novel method evaluation...")
    results = segmenter.evaluate_on_dataset(dataset_path, max_images=100)
    
    # Save results
    results_dir = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results/novel_method")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "novel_method_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n=== Novel Method Results (YOLO + SAM2) ===")
    for class_name in ['person', 'vehicle', 'overall']:
        if class_name in results:
            metrics = results[class_name]
            print(f"{class_name.upper()}:")
            print(f"  Mean Dice: {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f}")
            print(f"  Mean IoU:  {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
            print(f"  Median Dice: {metrics['median_dice']:.4f}")
            print(f"  Median IoU:  {metrics['median_iou']:.4f}")
            print(f"  Images: {metrics['num_images']}")
            print()
    
    # Compare with baseline
    baseline_file = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel/results/baseline/baseline_results.json")
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        print("\n=== Comparison with Baseline ===")
        for class_name in ['person', 'vehicle', 'overall']:
            if class_name in results and class_name in baseline_results:
                novel_dice = results[class_name]['mean_dice']
                baseline_dice = baseline_results[class_name]['mean_dice']
                novel_iou = results[class_name]['mean_iou']
                baseline_iou = baseline_results[class_name]['mean_iou']
                
                improvement_dice = ((novel_dice - baseline_dice) / baseline_dice * 100) if baseline_dice > 0 else float('inf')
                improvement_iou = ((novel_iou - baseline_iou) / baseline_iou * 100) if baseline_iou > 0 else float('inf')
                
                print(f"{class_name.upper()}:")
                print(f"  Dice improvement: {improvement_dice:.1f}% ({baseline_dice:.4f} → {novel_dice:.4f})")
                print(f"  IoU improvement:  {improvement_iou:.1f}% ({baseline_iou:.4f} → {novel_iou:.4f})")
                print()


if __name__ == "__main__":
    main()
