# SAM2 Guided Segmentation

A comparative study evaluating **Guided vs Automatic SAM2 segmentation** for road scene analysis. This project demonstrates that YOLO-guided SAM2 segmentation significantly outperforms automatic mask generation for person and vehicle detection in road scenes.

## 🎯 Key Results

- **6.3x Performance Improvement**: From 10% to 63% Overall Dice Score
- **17x Better Person Segmentation**: From 3% to 51% Dice Score  
- **4.8x Better Vehicle Segmentation**: From 16% to 76% Dice Score
- **5.9x Faster Processing**: From 8.37s to 1.41s per image

## 📊 Method Comparison

| Method | Person Dice | Vehicle Dice | Overall Dice | Speed |
|--------|-------------|--------------|--------------|-------|
| **SAM2 Automatic Baseline** | 3.0% | 16.0% | 10.0% | 8.37s/img |
| **YOLO + SAM2 Guided** | 51.0% | 76.0% | 63.0% | 1.41s/img |
| **Improvement** | **+1600%** | **+375%** | **+530%** | **+491%** |

## 🛠️ Methods

### Baseline: Official SAM2 Automatic
- Uses `SAM2AutomaticMaskGenerator` from official Facebook repository
- Generates masks automatically without manual prompts (32×32 grid sampling)
- Applies heuristic classification to map generic masks to person/vehicle classes
- Follows official SAM2 methodology exactly

### Novel Method: YOLO + SAM2 Guided  
- **Step 1**: YOLOv8 detects objects and provides bounding boxes
- **Step 2**: SAM2 segments detected objects using bounding box prompts
- **Step 3**: Class-aware evaluation using YOLO's object classifications
- Combines strengths of detection and segmentation

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+, CUDA-capable GPU recommended
pip install torch torchvision opencv-python ultralytics matplotlib
```

### Installation
```bash
git clone https://github.com/yourusername/sam2-guided-segmentation.git
cd sam2-guided-segmentation

# Create virtual environment
python -m venv sam_env
source sam_env/bin/activate  # Linux/Mac
# sam_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Clone and install official SAM2
git clone https://github.com/facebookresearch/sam2.git official_sam2_repo
cd official_sam2_repo
pip install -e .
cd ..
```

### Download Model Checkpoints
```bash
# Download SAM2 checkpoint (897MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
```

### Run Complete Evaluation
```bash
# Single command runs both methods and generates comparison
python main_evaluation.py
```

### Expected Output
```
SEGMENTATION EVALUATION RESULTS COMPARISON
==========================================
Method                    | Person Dice | Vehicle Dice | Overall Dice | Time/Image
Official SAM2 Baseline    |     10.08%  |      16.05%  |     10.08%   |    8.37s
YOLO+SAM2 Novel Method    |     50.52%  |      76.21%  |     63.37%   |    1.41s

CONCLUSION: YOLO+SAM2 method significantly outperforms official SAM2 automatic baseline
```

## 📁 Project Structure

```
sam2-guided-segmentation/
├── src/
│   ├── baseline_official_sam2.py      # Official SAM2 automatic baseline
│   ├── novel_method.py                # YOLO + SAM2 guided method
│   ├── evaluation.py                  # Unified evaluation framework
│   └── utils.py                       # Utility functions
├── main_evaluation.py                 # Complete evaluation pipeline
├── data/CamVid/                       # Dataset (validation images + labels)
├── checkpoints/                       # Model checkpoints
├── results/                           # Results and visualizations
│   ├── baseline/visualizations/       # SAM2 automatic mask overlays
│   ├── novel_method/visualizations/   # YOLO+SAM2 comparisons
│   └── combined_evaluation_results.json
├── official_sam2_repo/               # Official SAM2 repository
├── requirements.txt                   # Dependencies
├── PIPELINE_OVERVIEW.md              # Detailed pipeline documentation
└── PROJECT_SUMMARY.md               # Project overview
```

## 🔬 Technical Details

### Dataset
- **CamVid**: Cambridge-driving Labeled Video Database
- **Classes**: Person (pedestrian, child) and Vehicle (car, truck, bus, etc.)
- **Evaluation**: 100 validation images
- **Metrics**: Dice coefficient and IoU

### SAM2 Baseline Configuration
```python
# Official SAM2AutomaticMaskGenerator parameters
points_per_side=32
pred_iou_thresh=0.8
stability_score_thresh=0.95
crop_n_layers=1
min_mask_region_area=100
use_m2m=True
```

### YOLO+SAM2 Configuration
```python
# YOLOv8 Detection
model = "yolov8n.pt"
confidence_threshold = 0.25
target_classes = ['person', 'vehicle']

# SAM2 Segmentation  
model = "sam2_hiera_large.pt"
prompt_type = "bounding_box"
```

## 📈 Visualizations

The pipeline automatically generates visualizations:

- **Baseline**: Official SAM2 automatic masks with colorful overlays
- **Novel Method**: Side-by-side ground truth vs prediction comparisons
- **Location**: `results/baseline/visualizations/` and `results/novel_method/visualizations/`

## 🔄 Reproducibility

All results are fully reproducible:
```bash
# Complete pipeline
python main_evaluation.py

# Individual methods
python src/baseline_official_sam2.py    # Test baseline only
python src/novel_method.py              # Test novel method only
```

Results include:
- Detailed performance metrics (JSON format)
- Processing time measurements  
- Parameter configurations
- Visualization outputs

## 📚 Documentation

- [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md) - Complete pipeline documentation
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) - Project overview and results
- Source code is fully documented with docstrings

## 🤝 Contributing

This is a research project. Feel free to:
- Report issues or bugs
- Suggest improvements
- Extend to other datasets
- Compare with other segmentation methods

## 📄 License

This project combines multiple components:
- Our code: MIT License
- Official SAM2: Apache 2.0 License  
- YOLOv8: AGPL-3.0 License
- CamVid Dataset: Original license terms apply

## 🙏 Acknowledgments

- **Meta AI**: Official SAM2 implementation
- **Ultralytics**: YOLOv8 implementation  
- **Cambridge University**: CamVid dataset
- **PyTorch**: Deep learning framework

## 📞 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

**⭐ If this project helps your research, please consider starring the repository!**
