# Quick Start Guide

Get your gesture-controlled car running in 5 steps!

## ⚡ Super Quick Start (For Experienced Users)

```bash
# 1. Install
pip install torch torchvision opencv-python mediapipe pygame numpy Pillow

# 2. Prepare dataset (100+ images per class in 5 folders)
python prepare_dataset.py

# 3. Train
python train.py

# 4. Run
python main.py
```

## 📋 Step-by-Step Guide

### Step 1: Install Python (Skip if already installed)
- Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

### Step 2: Download Project
- Download and extract `gesture_car_project.zip`
- Open terminal/command prompt
- Navigate to project: `cd path/to/gesture_car_project`

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision opencv-python mediapipe pygame numpy Pillow
```

### Step 4: Prepare Dataset

**Option A - Use existing dataset:**
- Download a hand gesture dataset from Kaggle
- Extract to `./dataset/` folder
- Rename folders to: forward, backward, left, right, stop

**Option B - Create folder structure:**
```bash
python prepare_dataset.py
```
Then add your images to each folder (100+ per gesture recommended)

**Option C - Sample data (for testing only):**
If you just want to test the code without a full dataset:
1. Add 5-10 sample images to each gesture folder
2. Training accuracy will be low, but you can test the pipeline

### Step 5: Train Model
```bash
python train.py
```
- Wait for training to complete (10-30 mins with GPU, 1-2 hours without)
- Look for `model.pth` file when done
- Target accuracy: >80%

### Step 6: Run Simulation
```bash
python main.py
```
- Webcam window will open
- Show hand gestures to control the car
- Press Q to quit

## 🎮 Gesture Reference

| Make this gesture | Car does this |
|-------------------|---------------|
| Open palm facing up | Forward ⬆️ |
| Open palm facing down | Backward ⬇️ |
| Thumbs left | Left ⬅️ |
| Thumbs right | Right ➡️ |
| Closed fist | Stop ⏹️ |

*Note: Exact gestures depend on your training dataset*

## 🔧 Test Individual Components

**Test gesture detection only:**
```bash
python realtime_control.py
```

**Test car simulation only (keyboard controls):**
```bash
python car_simulation.py
# Use W/A/S/D keys
```

## ⚠️ Common Issues

**"No module named 'torch'"**
```bash
pip install torch torchvision
```

**"Dataset not found"**
- Check that `dataset/` folder exists
- Verify 5 subfolders: forward, backward, left, right, stop
- Run: `python prepare_dataset.py` (option 2 to validate)

**"Model file not found"**
- Train the model first: `python train.py`
- Make sure `model.pth` exists in project folder

**"Could not open webcam"**
- Close other apps using camera
- Check camera permissions
- Try different camera: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in realtime_control.py

**Low accuracy (<70%)**
- Collect more training images (aim for 100+ per class)
- Ensure good lighting in images
- Try pretrained model: set `USE_PRETRAINED = True` in train.py

## 💡 Pro Tips

1. **GPU Training**: Much faster if you have NVIDIA GPU
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`
   - If True, training will automatically use GPU

2. **Better Accuracy**:
   - Use 100+ images per gesture class
   - Vary hand positions, angles, lighting
   - Keep backgrounds simple

3. **Google Colab**: Use free GPU for training
   - See `GOOGLE_COLAB_SETUP.md` for instructions
   - Train in Colab, download model, run locally

4. **Debugging**:
   - Check each script individually first
   - Verify dataset with: `python prepare_dataset.py` (option 2)
   - Test model loading before running full simulation

## 📊 Expected Performance

**With good dataset (100+ images/class):**
- Training time: 10-30 minutes (GPU) or 1-2 hours (CPU)
- Training accuracy: 90-95%
- Validation accuracy: 80-90%
- Real-time FPS: 30-60 (smooth control)

**With minimal dataset (10 images/class):**
- Training time: 5-10 minutes
- Training accuracy: 70-80%
- Validation accuracy: 50-70%
- Real-time FPS: 30-60
- *Note: Low accuracy = unreliable gesture detection*

## 🎯 Minimum System Requirements

- **CPU**: Any modern processor (2+ cores recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Webcam**: Any USB or built-in webcam
- **OS**: Windows 10+, macOS 10.14+, or Linux

## 📚 Next Steps

After getting the basic simulation working:

1. **Improve accuracy**:
   - Collect more training data
   - Try pretrained ResNet18 model
   - Adjust hyperparameters in train.py

2. **Customize simulation**:
   - Change car speed in car_simulation.py
   - Modify window size
   - Add sound effects

3. **Extend functionality**:
   - Add more gestures
   - Multiple cars
   - Obstacles and scoring
   - Multiplayer mode

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Review `GOOGLE_COLAB_SETUP.md` for Colab instructions
3. Validate dataset: `python prepare_dataset.py`
4. Check training logs for errors

## ✅ Success Checklist

Before asking for help, verify:
- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip list`)
- [ ] Dataset organized correctly (5 folders with images)
- [ ] Model trained successfully (model.pth exists)
- [ ] Webcam working (test with camera app)
- [ ] No error messages during training

---

**You're ready! Have fun! 🎉**

Questions? Issues? Check the troubleshooting section in README.md