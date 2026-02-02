# Gesture-Controlled Virtual Car Simulation

A complete Python AI project that uses hand gesture recognition to control a virtual car in real-time. The project combines computer vision, deep learning, and game development.

## 🎯 Features

- **Real-time Hand Gesture Detection**: Uses MediaPipe and webcam for accurate hand tracking
- **Deep Learning Model**: Custom CNN or ResNet18 for gesture classification
- **5 Gesture Controls**: Forward, Backward, Left, Right, Stop
- **Virtual Car Simulation**: Pygame-based interactive car simulation
- **Complete Pipeline**: Training, inference, and visualization all included

## 📋 Requirements

### Software Requirements
- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster training)

### Python Libraries
```bash
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
mediapipe>=0.10.0
pygame>=2.5.0
numpy>=1.24.0
Pillow>=10.0.0
```

## 📁 Project Structure

```
gesture_car_project/
├── dataset/                  # Dataset organized by gesture class
│   ├── forward/             # Forward gesture images
│   ├── backward/            # Backward gesture images
│   ├── left/                # Left gesture images
│   ├── right/               # Right gesture images
│   └── stop/                # Stop gesture images
├── model.py                 # CNN model architecture
├── train.py                 # Model training script
├── realtime_control.py      # Webcam gesture detection
├── car_simulation.py        # Pygame car simulation
├── main.py                  # Integrated simulation
├── prepare_dataset.py       # Dataset organization tool
├── model.pth               # Trained model weights (generated)
├── class_names.json        # Class label mapping (generated)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd gesture_car_project

# Install dependencies
pip install torch torchvision opencv-python mediapipe pygame numpy Pillow
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

You need a dataset of hand gesture images organized into 5 folders. You have several options:

#### Option A: Use the Dataset Preparation Tool
```bash
python prepare_dataset.py
```

This interactive tool helps you:
- Create the proper folder structure
- Validate your dataset
- Copy images from other sources
- Get dataset download instructions

#### Option B: Manual Organization
Create the following structure:
```
dataset/
├── forward/     # 100+ images of forward gesture
├── backward/    # 100+ images of backward gesture
├── left/        # 100+ images of left gesture
├── right/       # 100+ images of right gesture
└── stop/        # 100+ images of stop gesture
```

#### Option C: Download Existing Dataset
- Search Kaggle for "hand gesture dataset"
- Download and extract to `./dataset/`
- Rename folders to match: forward, backward, left, right, stop

**Recommended**: 100+ images per gesture class (500+ total)

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess your dataset
- Train a CNN model for gesture recognition
- Validate on 20% of data
- Save the best model as `model.pth`
- Save class names as `class_names.json`

**Training Parameters** (configurable in `train.py`):
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001
- Image size: 224x224
- Train/Val split: 80/20

**Expected Results**:
- Training accuracy: >90%
- Validation accuracy: >80%
- Training time: 10-30 minutes (GPU) or 1-2 hours (CPU)

### 4. Test Individual Components

#### Test Gesture Detection Only
```bash
python realtime_control.py
```
This opens your webcam and displays detected gestures in real-time.

#### Test Car Simulation Only
```bash
python car_simulation.py
```
This runs the car simulation with keyboard controls (W/A/S/D).

### 5. Run Complete Simulation

```bash
python main.py
```

This integrates everything:
1. Captures webcam feed
2. Detects hand gestures
3. Controls virtual car in Pygame window
4. Shows real-time gesture and statistics

**Controls**:
- Show hand gestures to camera
- Press `Q` or `ESC` to quit

## 🎮 Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| Forward | ⬆️ Move Up | Car moves upward on screen |
| Backward | ⬇️ Move Down | Car moves downward on screen |
| Left | ⬅️ Move Left | Car moves left on screen |
| Right | ➡️ Move Right | Car moves right on screen |
| Stop | ⏹️ Stop | Car stops moving |

## 🔧 Configuration

### Model Architecture

**Default**: Custom CNN (GestureCNN)
- 4 Convolutional blocks with BatchNorm
- 2 Fully connected layers with Dropout
- ~13M parameters

**Alternative**: ResNet18 (Pretrained)
- Transfer learning from ImageNet
- ~11M parameters
- Often faster to train and more accurate

To use ResNet18, set in `train.py`:
```python
USE_PRETRAINED = True
```

### Hyperparameters

Edit these in `train.py`:
```python
BATCH_SIZE = 32          # Batch size for training
NUM_EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.001    # Learning rate
IMG_SIZE = 224           # Input image size
TRAIN_SPLIT = 0.8        # Train/validation split
```

### Simulation Settings

Edit these in `car_simulation.py`:
```python
width = 800              # Window width
height = 600             # Window height
fps = 60                 # Frame rate
car.speed = 5            # Car movement speed
```

## 🐛 Troubleshooting

### Dataset Issues

**Problem**: "ERROR: Dataset not found"
```bash
# Verify dataset structure
python prepare_dataset.py
# Select option 2 to validate
```

**Problem**: Low accuracy (<70%)
- Collect more images (aim for 100+ per class)
- Ensure good lighting when capturing images
- Use diverse hand positions and angles
- Try using pretrained ResNet18

### Webcam Issues

**Problem**: "Could not open webcam"
- Check camera permissions
- Ensure no other app is using camera
- Try different camera index: `cv2.VideoCapture(1)`

**Problem**: Poor gesture detection
- Ensure good lighting
- Keep hand in frame
- Maintain consistent distance from camera
- Avoid cluttered backgrounds

### Performance Issues

**Problem**: Slow training
- Use GPU if available (CUDA or MPS)
- Reduce batch size
- Use smaller model or pretrained ResNet18

**Problem**: Laggy simulation
- Reduce FPS in car_simulation.py
- Close other applications
- Use faster device

### Import Errors

**Problem**: "No module named..."
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 💡 Tips for Best Results

### Dataset Collection
1. **Diversity**: Vary hand positions, angles, and lighting
2. **Consistency**: Use similar backgrounds for each gesture class
3. **Quality**: Use clear, focused images
4. **Quantity**: More data = better accuracy (aim for 100+ per class)

### Gesture Recognition
1. **Lighting**: Use good, consistent lighting
2. **Background**: Keep background simple and consistent
3. **Distance**: Maintain consistent distance from camera
4. **Gestures**: Make clear, distinct gestures

### Training
1. **GPU**: Use GPU for 10-20x faster training
2. **Patience**: Let model train for full 20 epochs
3. **Validation**: Monitor validation accuracy to avoid overfitting
4. **Augmentation**: Data augmentation helps with small datasets

## 📊 Technical Details

### Model Architecture (GestureCNN)

```
Input: 224x224 RGB Image
├── Conv Block 1: 3→32 channels
├── Conv Block 2: 32→64 channels
├── Conv Block 3: 64→128 channels
├── Conv Block 4: 128→256 channels
├── Flatten: 256×14×14 = 50176
├── FC Layer 1: 50176→512
├── Dropout: 0.5
├── FC Layer 2: 512→128
├── Dropout: 0.5
└── FC Layer 3: 128→5 (classes)
```

### Data Preprocessing

**Training Augmentation**:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Resize to 224×224
- Normalize (ImageNet statistics)

**Validation/Inference**:
- Resize to 224×224
- Normalize (ImageNet statistics)

### Gesture Detection Pipeline

1. **Capture Frame**: OpenCV captures webcam frame
2. **Hand Detection**: MediaPipe detects hand landmarks
3. **Region Extraction**: Crop hand region with padding
4. **Preprocessing**: Resize, normalize, convert to tensor
5. **Prediction**: CNN predicts gesture class
6. **Smoothing**: Average over recent frames
7. **Command**: Map gesture to car control

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision**: Hand detection with MediaPipe
- **Deep Learning**: CNN architecture and training
- **PyTorch**: Model building, training, inference
- **Image Processing**: OpenCV for real-time video
- **Game Development**: Pygame for interactive simulation
- **Integration**: Combining multiple technologies

## 🔮 Future Enhancements

Potential improvements:
- [ ] Add more gestures (pinch, thumbs up, etc.)
- [ ] Multiple car support
- [ ] Obstacles and collision detection
- [ ] Score system and leaderboard
- [ ] Mobile app deployment
- [ ] 3D car model rendering
- [ ] Record and replay functionality
- [ ] Multi-player support

## 📚 Learning Resources

- **PyTorch**: https://pytorch.org/tutorials/
- **MediaPipe**: https://google.github.io/mediapipe/
- **OpenCV**: https://docs.opencv.org/
- **Pygame**: https://www.pygame.org/docs/

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better gesture recognition models
- Additional gestures
- Enhanced car simulation features
- Performance optimizations
- Documentation improvements

## 📄 License

This project is provided for educational purposes.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand detection
- **PyTorch** team for the deep learning framework
- **OpenCV** community for computer vision tools
- **Pygame** developers for game development framework

## 📧 Support

For issues and questions:
1. Check troubleshooting section
2. Validate dataset structure
3. Check model training logs
4. Verify all dependencies are installed

## 🎉 Success Checklist

- [ ] Dataset organized (5 folders, 100+ images each)
- [ ] Dependencies installed
- [ ] Model trained (validation accuracy >80%)
- [ ] model.pth file created
- [ ] Webcam working
- [ ] Gesture detection accurate
- [ ] Car simulation responsive
- [ ] Integrated system working

---

**Happy Coding! 🚗💨**

Have fun controlling your virtual car with hand gestures!