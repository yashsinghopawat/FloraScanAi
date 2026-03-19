"""
Dataset Download and Preparation Script
Downloads PlantVillage dataset and organizes it for training
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_directory_structure():
    """
    Create the required directory structure
    """
    directories = [
        'PlantVillage/train',
        'PlantVillage/val',
        'PlantVillage/test'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

def download_dataset():
    """
    Instructions to download the PlantVillage dataset
    """
    print("\n" + "="*60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\nThe PlantVillage dataset is available on Kaggle.")
    print("\n📥 OPTION 1: Download via Kaggle Website")
    print("-" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Run this script again to organize the data")
    
    print("\n📥 OPTION 2: Download via Kaggle API (Recommended)")
    print("-" * 60)
    print("1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print("\n2. Get Kaggle API credentials:")
    print("   - Go to kaggle.com/account")
    print("   - Click 'Create New API Token'")
    print("   - Download kaggle.json")
    print("   - Place it in: ~/.kaggle/kaggle.json (Linux/Mac)")
    print("                  C:\\Users\\<YourUser>\\.kaggle\\kaggle.json (Windows)")
    print("\n3. Run these commands:")
    print("   kaggle datasets download -d emmarex/plantdisease")
    print("   unzip plantdisease.zip -d PlantVillage_raw")
    
    print("\n📥 OPTION 3: Alternative Dataset")
    print("-" * 60)
    print("You can also use:")
    print("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("\nThis version is already split into train/val/test!")
    
    print("\n" + "="*60)

def organize_dataset(source_dir='PlantVillage_raw'):
    """
    Organize downloaded dataset into train/val/test splits
    """
    if not os.path.exists(source_dir):
        print(f"\n❌ Error: Source directory '{source_dir}' not found!")
        print("Please download the dataset first.")
        return False
    
    print("\n📂 Organizing dataset...")
    
    # If using the pre-split dataset
    if os.path.exists(os.path.join(source_dir, 'train')):
        print("✓ Found pre-split dataset!")
        
        # Copy to PlantVillage directory
        for split in ['train', 'val', 'test']:
            src = os.path.join(source_dir, split)
            dst = os.path.join('PlantVillage', split)
            
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"✓ Copied {split} split")
        
        print("\n✅ Dataset organized successfully!")
        return True
    
    # If dataset needs to be split
    else:
        print("Dataset needs to be split into train/val/test...")
        print("This will be done automatically during organization.")
        
        # Import splitfolders for splitting
        try:
            import splitfolders
        except ImportError:
            print("\n📦 Installing split-folders package...")
            os.system('pip install split-folders')
            import splitfolders
        
        # Split the dataset (70% train, 15% val, 15% test)
        print("\nSplitting dataset (70% train, 15% val, 15% test)...")
        splitfolders.ratio(
            source_dir,
            output='PlantVillage',
            seed=42,
            ratio=(0.7, 0.15, 0.15),
            group_prefix=None
        )
        
        print("\n✅ Dataset split successfully!")
        return True

def verify_dataset():
    """
    Verify dataset structure and count images
    """
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join('PlantVillage', split)
        
        if not os.path.exists(split_path):
            print(f"\n❌ {split.upper()} split not found!")
            continue
        
        # Count classes and images
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        total_images = 0
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_images += num_images
        
        print(f"\n{split.upper()} SET:")
        print(f"  Classes: {len(classes)}")
        print(f"  Total Images: {total_images}")
        
        # Show first 5 classes
        if classes:
            print(f"  Sample classes:")
            for cls in sorted(classes)[:5]:
                print(f"    - {cls}")
            if len(classes) > 5:
                print(f"    ... and {len(classes)-5} more")
    
    print("\n" + "="*60)

def main():
    """
    Main function to setup dataset
    """
    print("\n" + "="*60)
    print("PLANT DISEASE DATASET SETUP")
    print("="*60)
    
    # Check if dataset already exists
    if os.path.exists('PlantVillage/train') and os.path.exists('PlantVillage/val'):
        print("\n✓ Dataset already exists!")
        verify_dataset()
        
        response = input("\nDo you want to re-download/reorganize? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Create directory structure
    print("\n[Step 1/3] Creating directories...")
    create_directory_structure()
    
    # Show download instructions
    print("\n[Step 2/3] Dataset Download")
    download_dataset()
    
    # Wait for user to download
    print("\n" + "="*60)
    input("\nPress ENTER after you have downloaded the dataset...")
    
    # Organize dataset
    print("\n[Step 3/3] Organizing dataset...")
    
    # Check common directory names
    possible_dirs = ['PlantVillage_raw', 'plantdisease', 'PlantVillage', 'plant-disease']
    source_dir = None
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            source_dir = dir_name
            break
    
    if source_dir:
        organize_dataset(source_dir)
        verify_dataset()
    else:
        print("\n❌ Could not find downloaded dataset!")
        print("Please ensure the dataset is extracted in the current directory.")
        print(f"Looking for: {', '.join(possible_dirs)}")
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python train_model.py (to train the model)")
    print("2. Run: python predict.py <image_path> (to test predictions)")
    print("3. Run: streamlit run app.py (to launch web app)")

if __name__ == "__main__":
    main()
