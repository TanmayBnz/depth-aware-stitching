"""
Script to download and setup UDIS-D dataset
UDIS-D: Unsupervised Deep Image Stitching Dataset
Paper: https://github.com/nie-lang/UnsupervisedDeepImageStitching
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import shutil


def download_udis_d(data_dir="data/raw/dataset/UDIS-D"):
    """
    Download UDIS-D dataset
    
    Dataset structure:
    UDIS-D/
        testing/
            input1/  # First images
            input2/  # Second images  
            label/   # Ground truth panoramas
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UDIS-D Dataset Setup")
    print("=" * 60)
    
    # Note: UDIS-D needs to be downloaded manually from Google Drive
    print("\n⚠️  UDIS-D must be downloaded manually:")
    print("\n1. Visit: https://github.com/nie-lang/UnsupervisedDeepImageStitching")
    print("2. Download UDIS-D.zip from the Google Drive link")
    print("3. Extract to:", data_path.absolute())
    print("\nExpected structure:")
    print(f"{data_dir}/")
    print("    testing/")
    print("        input1/")
    print("        input2/")
    print("        label/")
    
    # Check if already downloaded
    test_path = data_path / "testing"
    if test_path.exists():
        input1_path = test_path / "input1"
        input2_path = test_path / "input2"
        
        if input1_path.exists() and input2_path.exists():
            n_pairs = len(list(input1_path.glob("*.jpg")))
            print(f"\n✅ Dataset found! {n_pairs} image pairs detected.")
            return True
    
    print(f"\n❌ Dataset not found at {data_path}")
    print("Please download manually following the instructions above.")
    return False


def verify_udis_structure(data_dir="data/raw/dataset/UDIS-D"):
    """
    Verify UDIS-D dataset structure and list available pairs
    """
    data_path = Path(data_dir)
    test_path = data_path / "testing"
    
    if not test_path.exists():
        print(f"❌ Testing directory not found: {test_path}")
        return False
    
    input1_path = test_path / "input1"
    input2_path = test_path / "input2"
    label_path = test_path / "label"
    
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    # Check directories
    for path, name in [(input1_path, "input1"), (input2_path, "input2"), (label_path, "label")]:
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"✅ {name}: {count} images")
        else:
            print(f"❌ {name}: Directory not found")
    
    # List some example pairs
    if input1_path.exists() and input2_path.exists():
        images1 = sorted(input1_path.glob("*.jpg"))
        images2 = sorted(input2_path.glob("*.jpg"))
        
        print(f"\n📋 Available image pairs ({len(images1)} total):")
        for i, (img1, img2) in enumerate(zip(images1[:5], images2[:5])):
            print(f"  {i+1}. {img1.name} + {img2.name}")
        
        if len(images1) > 5:
            print(f"  ... and {len(images1) - 5} more pairs")
        
        return True
    
    return False


def get_udis_pairs(data_dir="data/raw/dataset/UDIS-D", split="testing"):
    """
    Get list of image pairs from UDIS-D dataset
    
    Returns:
        list of tuples: [(img1_path, img2_path, label_path), ...]
    """
    data_path = Path(data_dir)
    split_path = data_path / split
    
    input1_path = split_path / "input1"
    input2_path = split_path / "input2"
    label_path = split_path / "label"
    
    if not (input1_path.exists() and input2_path.exists()):
        return []
    
    # Get all image pairs
    images1 = sorted(input1_path.glob("*.jpg"))
    images2 = sorted(input2_path.glob("*.jpg"))
    
    pairs = []
    for img1, img2 in zip(images1, images2):
        # Find corresponding label if exists
        label_file = label_path / img1.name if label_path.exists() else None
        if label_file and not label_file.exists():
            label_file = None
        
        pairs.append((str(img1), str(img2), str(label_file) if label_file else None))
    
    return pairs


def create_sample_pair_file(data_dir="data/raw/dataset/UDIS-D", output="udis_pairs.txt"):
    """
    Create a text file listing all UDIS-D pairs for batch testing
    """
    pairs = get_udis_pairs(data_dir)
    
    if not pairs:
        print("❌ No pairs found")
        return False
    
    output_path = Path(output)
    with open(output_path, 'w') as f:
        f.write("# UDIS-D Dataset Pairs\n")
        f.write("# Format: img1_path,img2_path,label_path\n\n")
        
        for img1, img2, label in pairs:
            label_str = label if label else "None"
            f.write(f"{img1},{img2},{label_str}\n")
    
    print(f"✅ Created pair list: {output_path} ({len(pairs)} pairs)")
    return True


if __name__ == "__main__":
    # Try to download/setup
    download_udis_d()
    
    # Verify structure
    if verify_udis_structure():
        # Create pair list file
        create_sample_pair_file()
    else:
        print("\n💡 Please download the dataset manually and run this script again.")