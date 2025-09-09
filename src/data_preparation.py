#!/usr/bin/env python3
"""
Data preparation script for CamVid dataset
"""
import os
import kaggle
import zipfile
from pathlib import Path

def download_camvid_dataset():
    """Download CamVid dataset from Kaggle"""
    
    # Set up paths
    project_root = Path("/anvil/projects/x-soc250046/x-sishraq/SegmentAnythingModel")
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Download dataset
    print("Downloading CamVid dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        'carlolepelaars/camvid', 
        path=str(data_dir), 
        unzip=True
    )
    
    print("Dataset downloaded successfully!")
    
    # Check structure
    camvid_dir = data_dir / "camvid"
    if camvid_dir.exists():
        print(f"Dataset structure:")
        for item in camvid_dir.iterdir():
            print(f"  {item.name}")
    
    return camvid_dir

def analyze_dataset(camvid_dir):
    """Analyze the dataset structure and validation set"""
    
    val_dir = camvid_dir / "val"
    if val_dir.exists():
        print(f"\nValidation set analysis:")
        print(f"Images: {len(list((val_dir / 'images').glob('*')))}")
        print(f"Labels: {len(list((val_dir / 'labels').glob('*')))}")
    
    # Check class labels
    class_file = camvid_dir / "class_dict.csv"
    if class_file.exists():
        import pandas as pd
        classes = pd.read_csv(class_file)
        print(f"\nAvailable classes:")
        print(classes)
        
        # Find person and vehicle classes
        person_classes = classes[classes['name'].str.contains('person|pedestrian', case=False)]
        vehicle_classes = classes[classes['name'].str.contains('car|vehicle|truck|bus|bike|motorcycle', case=False)]
        
        print(f"\nPerson-related classes:")
        print(person_classes)
        print(f"\nVehicle-related classes:")
        print(vehicle_classes)

if __name__ == "__main__":
    # Make sure Kaggle credentials are set up
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("Please set up your Kaggle credentials first:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create new API token")
        print("3. Save kaggle.json to ~/.kaggle/")
        exit(1)
    
    # Download and analyze dataset
    camvid_dir = download_camvid_dataset()
    analyze_dataset(camvid_dir)
