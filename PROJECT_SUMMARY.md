# SAM2 Assignment - Clean Implementation

## Project Overview
This project implements and compares two approaches for automatic segmentation of people and vehicles in road scenes:

1. **Baseline**: Official SAM2 Automatic Mask Generator (from facebook/sam2 repository)
2. **Novel Method**: YOLO object detection + SAM2 segmentation

## Implementation Files

### Core Implementation
- `src/baseline_official_sam2.py` - Official SAM2 automatic baseline using SAM2AutomaticMaskGenerator
- `src/novel_method.py` - YOLO + SAM2 novel method
- `src/evaluation.py` - Evaluation framework
- `src/utils.py` - Utility functions

### Main Scripts
- `main_evaluation.py` - Complete evaluation comparing both methods
- `data_preparation.py` - Dataset preparation (if needed)

### Data
- `data/CamVid/val/` - Validation images
- `data/CamVid/val_labels/` - Ground truth masks

## Results

### Latest Evaluation Results
- **Official SAM2 Baseline**: 10.08% Overall Dice
- **YOLO+SAM2 Novel Method**: 63.37% Overall Dice
- **Improvement**: 528.6% (6.3x better)

### Key Achievements
✅ Proper baseline using official SAM2 repository methodology  
✅ No manual prompts required - fully automatic  
✅ Novel method significantly outperforms baseline  
✅ Comprehensive evaluation with 2 decimal place precision  

## Usage

```bash
# Run complete evaluation
python main_evaluation.py

# Test baseline only
python src/baseline_official_sam2.py

# Test novel method only
python src/novel_method.py
```

## Method Details

### Baseline (Official SAM2)
- Uses `SAM2AutomaticMaskGenerator` from official repository
- Generates masks automatically without manual prompts
- Uses heuristics to classify masks as person/vehicle based on size/position
- Follows official SAM2 methodology

### Novel Method (YOLO + SAM2)
- YOLOv8 detects objects (person/vehicle) with bounding boxes
- SAM2 segments detected objects using bounding box prompts
- Combines strengths of object detection and segmentation
- Achieves superior performance through guided segmentation

## Dataset
- **CamVid**: Cambridge-driving Labeled Video Database
- **Classes**: Person (pedestrian, child) and Vehicle (car, truck, bus, etc.)
- **Evaluation**: Dice coefficient and IoU metrics
- **Images**: 100 validation images for evaluation

## Project Objective

Provide a clear, fair, and reproducible comparison between:
1. A faithful, fully automatic SAM2 baseline (no manual prompts, class-agnostic generation + heuristic class mapping), and
2. A guided segmentation pipeline (YOLO detections as box prompts into SAM2) that leverages class-aware localization to improve mask quality for people and vehicles in road scenes.

The objective is to quantify the performance gap, analyze contributing factors, and document how prompt guidance changes segmentation effectiveness relative to pure automatic mask proposals.

## Required Tasks

- Baseline Integration: Load official SAM2 checkpoint and run `SAM2AutomaticMaskGenerator` with documented parameters.
- Heuristic Classification: Map generic masks to person/vehicle via geometric and positional heuristics; exclude ambiguous masks.
- Guided Pipeline: Run YOLOv8 detection, filter to target classes, feed bounding boxes into SAM2 predictor for per-object masks.
- Unified Evaluation: Compute Dice and IoU per class and overall; save results to structured JSON (`combined_evaluation_results.json`).
- Parameter Logging: Persist baseline generator parameter set in results metadata for transparency.
- Fairness Assessment: Record differences in prior knowledge (detection vs heuristic) and cite potential strengthened baseline variants (higher grid density, multi-scale crops, looser thresholds).
- Reproducibility: Single entry script (`main_evaluation.py`) to regenerate all reported metrics.
- Reporting: Maintain this summary file with up-to-date objective, tasks, and headline metrics.

Nice-to-Have (Deferred):
- Enhanced Baseline Variant (multi-scale + tuned thresholds) for robustness check.
- Runtime profiling (ms per image) with and without GPU warm-up.
- Additional qualitative visualization gallery.
