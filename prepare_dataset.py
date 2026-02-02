"""
prepare_dataset.py - Dataset Organization Helper

This script helps organize gesture images into proper folders
for training. It can:
- Create folder structure
- Copy images from a source directory
- Validate dataset structure
"""

import os
import shutil
from pathlib import Path


class DatasetOrganizer:
    """
    Helper class to organize gesture dataset
    """
    
    def __init__(self, dataset_path='./dataset'):
        """
        Initialize dataset organizer
        
        Args:
            dataset_path (str): Path to dataset directory
        """
        self.dataset_path = dataset_path
        self.gesture_classes = ['forward', 'backward', 'left', 'right', 'stop']
    
    def create_structure(self):
        """
        Create dataset folder structure
        """
        print(f"Creating dataset structure at: {self.dataset_path}")
        
        # Create main dataset directory
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Create subdirectories for each gesture class
        for gesture_class in self.gesture_classes:
            class_path = os.path.join(self.dataset_path, gesture_class)
            os.makedirs(class_path, exist_ok=True)
            print(f"  ✓ Created: {class_path}")
        
        print("\n✓ Dataset structure created successfully!")
        self.print_structure()
    
    def print_structure(self):
        """
        Print expected dataset structure
        """
        print("\nExpected structure:")
        print(f"{self.dataset_path}/")
        for gesture_class in self.gesture_classes:
            print(f"  ├── {gesture_class}/")
            print(f"  │   ├── image1.jpg")
            print(f"  │   ├── image2.jpg")
            print(f"  │   └── ...")
    
    def validate_dataset(self):
        """
        Validate dataset structure and count images
        
        Returns:
            bool: Whether dataset is valid
        """
        print(f"\nValidating dataset at: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return False
        
        valid = True
        total_images = 0
        
        print("\nClass distribution:")
        for gesture_class in self.gesture_classes:
            class_path = os.path.join(self.dataset_path, gesture_class)
            
            if not os.path.exists(class_path):
                print(f"  ❌ Missing folder: {gesture_class}")
                valid = False
                continue
            
            # Count images in folder
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            num_images = len(image_files)
            total_images += num_images
            
            if num_images == 0:
                print(f"  ⚠️  {gesture_class}: {num_images} images (EMPTY!)")
                valid = False
            elif num_images < 50:
                print(f"  ⚠️  {gesture_class}: {num_images} images (recommended: 100+)")
            else:
                print(f"  ✓ {gesture_class}: {num_images} images")
        
        print(f"\nTotal images: {total_images}")
        
        if total_images < 250:
            print("⚠️  Warning: Dataset is small. Recommended: 500+ images total")
            print("   Consider collecting more images for better accuracy")
        
        if valid:
            print("\n✓ Dataset structure is valid")
        else:
            print("\n❌ Dataset validation failed")
            print("   Please ensure all 5 gesture folders exist and contain images")
        
        return valid
    
    def copy_images_from_source(self, source_dir, class_mapping=None):
        """
        Copy images from a source directory to organized structure
        
        Args:
            source_dir (str): Source directory containing images
            class_mapping (dict): Optional mapping from source folders to gesture classes
                Example: {'fist': 'stop', 'palm': 'forward'}
        """
        print(f"\nCopying images from: {source_dir}")
        
        if not os.path.exists(source_dir):
            print(f"❌ Source directory does not exist: {source_dir}")
            return
        
        # If no mapping provided, assume source has same structure
        if class_mapping is None:
            class_mapping = {cls: cls for cls in self.gesture_classes}
        
        copied_count = 0
        
        for source_class, target_class in class_mapping.items():
            source_path = os.path.join(source_dir, source_class)
            target_path = os.path.join(self.dataset_path, target_class)
            
            if not os.path.exists(source_path):
                print(f"  ⚠️  Source folder not found: {source_class}")
                continue
            
            # Create target directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            
            # Copy all image files
            image_files = [f for f in os.listdir(source_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                source_file = os.path.join(source_path, img_file)
                target_file = os.path.join(target_path, img_file)
                
                try:
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                except Exception as e:
                    print(f"  ❌ Error copying {img_file}: {e}")
            
            print(f"  ✓ Copied {len(image_files)} images: {source_class} -> {target_class}")
        
        print(f"\n✓ Total images copied: {copied_count}")
    
    def download_sample_dataset(self):
        """
        Provide instructions for downloading sample dataset
        """
        print("\n" + "="*60)
        print("DATASET DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("\nTo train the model, you need a gesture dataset.")
        print("Here are some options:")
        print("\n1. Create your own dataset:")
        print("   - Use your webcam to capture images")
        print("   - Make 5 different hand gestures (100+ images each)")
        print("   - Organize into folders: forward, backward, left, right, stop")
        
        print("\n2. Use existing hand gesture datasets:")
        print("   - Kaggle: 'Hand Gesture Recognition Database'")
        print("   - Search: 'hand gesture dataset' on Kaggle or GitHub")
        print("   - Download and extract to ./dataset/")
        
        print("\n3. Google Colab users:")
        print("   - Mount Google Drive:")
        print("     from google.colab import drive")
        print("     drive.mount('/content/drive')")
        print("   - Organize dataset in Drive:")
        print("     /content/drive/MyDrive/gesture_dataset/")
        print("   - Update DATASET_PATH in train.py accordingly")
        
        print("\n4. Recommended dataset structure:")
        self.print_structure()
        
        print("\n" + "="*60)


def main():
    """
    Main function with interactive menu
    """
    print("="*60)
    print("GESTURE DATASET PREPARATION TOOL")
    print("="*60)
    
    organizer = DatasetOrganizer(dataset_path='./dataset')
    
    while True:
        print("\nOptions:")
        print("  1. Create dataset folder structure")
        print("  2. Validate existing dataset")
        print("  3. Copy images from source directory")
        print("  4. Show dataset download instructions")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            organizer.create_structure()
        
        elif choice == '2':
            organizer.validate_dataset()
        
        elif choice == '3':
            source = input("Enter source directory path: ").strip()
            
            print("\nDo you need custom class mapping?")
            print("(If your source folders have different names)")
            use_mapping = input("Use custom mapping? (y/n): ").strip().lower()
            
            if use_mapping == 'y':
                print("\nEnter mapping (source_folder:target_gesture)")
                print("Example: fist:stop, palm:forward")
                mapping_str = input("Mapping: ").strip()
                
                # Parse mapping
                class_mapping = {}
                for pair in mapping_str.split(','):
                    if ':' in pair:
                        src, tgt = pair.strip().split(':')
                        class_mapping[src.strip()] = tgt.strip()
                
                organizer.copy_images_from_source(source, class_mapping)
            else:
                organizer.copy_images_from_source(source)
        
        elif choice == '4':
            organizer.download_sample_dataset()
        
        elif choice == '5':
            print("\nExiting...")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()