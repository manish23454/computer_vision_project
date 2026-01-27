#!/usr/bin/env python3
"""
Setup Script for Vision Tracking System
Checks dependencies and creates necessary folders
"""

import sys
import os
import subprocess


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"✗ Python 3.7+ required (found {version.major}.{version.minor})")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'serial': 'pyserial'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (not installed)")
            missing_packages.append(package_name)
    
    return missing_packages


def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--user'
        ] + packages)
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install packages")
        print("  Please install manually:")
        print(f"  pip install {' '.join(packages)}")
        return False


def create_folders():
    """Create necessary folders"""
    print("\nCreating folders...")
    
    folders = [
        'face_dataset',
        'debug_frames'
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✓ Created: {folder}/")
        else:
            print(f"✓ Exists: {folder}/")


def test_camera():
    """Test if camera is accessible"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        
        if camera.isOpened():
            ret, frame = camera.read()
            camera.release()
            
            if ret:
                print("✓ Camera accessible")
                return True
            else:
                print("⚠ Camera opened but cannot read frames")
                return False
        else:
            print("⚠ Cannot open camera")
            print("  Check if camera is connected and not in use")
            return False
    
    except Exception as e:
        print(f"⚠ Camera test failed: {e}")
        return False


def create_sample_config():
    """Check if config.py exists"""
    if os.path.exists('config.py'):
        print("\n✓ config.py exists")
        return True
    else:
        print("\n✗ config.py not found")
        return False


def main():
    """Run setup checks"""
    print("="*60)
    print("Vision Tracking System - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print("\n" + "="*60)
        print("Missing Dependencies Detected")
        print("="*60)
        
        response = input("\nInstall missing packages automatically? (y/n): ")
        
        if response.lower() == 'y':
            if not install_dependencies(missing):
                sys.exit(1)
        else:
            print("\nPlease install manually:")
            print(f"pip install {' '.join(missing)}")
            sys.exit(1)
    
    # Create folders
    create_folders()
    
    # Check config
    if not create_sample_config():
        sys.exit(1)
    
    # Test camera (optional)
    test_camera()
    
    # Final summary
    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and customize config.py")
    print("2. (Optional) Create face dataset: python dataset_creator.py")
    print("3. Run test suite: python test_system.py")
    print("4. Start tracking system: python vision_system.py")
    print("\nFor Arduino integration:")
    print("5. Upload arduino_receiver.ino to Arduino")
    print("6. Set ENABLE_SERIAL = True in config.py")
    print("="*60)


if __name__ == "__main__":
    main()