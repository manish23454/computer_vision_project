# Google Colab Setup Instructions

This guide helps you run the Gesture-Controlled Virtual Car Simulation in Google Colab.

## 📌 Important Notes

- Google Colab provides free GPU access for faster training
- Webcam access in Colab is limited - you'll need to use local Python for real-time gesture detection
- Use Colab for **model training only**, then download the model for local use

## 🚀 Setup Steps

### 1. Upload Project to Google Drive

First, organize your project in Google Drive:

```
MyDrive/
└── gesture_car_project/
    ├── dataset/
    │   ├── forward/
    │   ├── backward/
    │   ├── left/
    │   ├── right/
    │   └── stop/
    ├── model.py
    ├── train.py
    └── requirements.txt
```

### 2. Create a New Colab Notebook

Create a new notebook in Google Colab and follow these steps:

#### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link and authorize access.

#### Step 2: Navigate to Project Directory

```python
import os
os.chdir('/content/drive/MyDrive/gesture_car_project')
!pwd
!ls
```

#### Step 3: Install Dependencies

```python
!pip install torch torchvision opencv-python mediapipe pygame numpy Pillow
```

#### Step 4: Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

To enable GPU:
- Go to **Runtime → Change runtime type**
- Select **GPU** as Hardware accelerator
- Click **Save**

#### Step 5: Verify Dataset

```python
!python prepare_dataset.py
# Select option 2 to validate dataset
```

Or manually check:

```python
import os
dataset_path = './dataset'
classes = ['forward', 'backward', 'left', 'right', 'stop']

for cls in classes:
    path = os.path.join(dataset_path, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{cls}: {count} images")
    else:
        print(f"{cls}: folder not found!")
```

#### Step 6: Train the Model

```python
!python train.py
```

This will:
- Train the model using GPU (much faster!)
- Save `model.pth` and `class_names.json`
- Display training progress and accuracy

Training typically takes:
- With GPU: 10-20 minutes
- Without GPU: 1-2 hours

#### Step 7: Download Trained Model

After training completes, download the model files:

```python
from google.colab import files

# Download model weights
files.download('model.pth')

# Download class names
files.download('class_names.json')
```

### 3. Use Model Locally

Once you've downloaded `model.pth` and `class_names.json`:

1. Copy them to your local project directory
2. Run the simulation on your local machine:

```bash
python main.py
```

## 📊 Alternative: Complete Colab Notebook

Here's a complete notebook template:

```python
# ===== CELL 1: Setup =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 2: Navigate and Install =====
import os
os.chdir('/content/drive/MyDrive/gesture_car_project')
!pip install -q torch torchvision opencv-python mediapipe pygame

# ===== CELL 3: Check GPU =====
import torch
print(f"Using: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ===== CELL 4: Validate Dataset =====
!python prepare_dataset.py
# Select option 2

# ===== CELL 5: Train Model =====
!python train.py

# ===== CELL 6: Download Model =====
from google.colab import files
files.download('model.pth')
files.download('class_names.json')
```

## 🎯 Dataset Options for Colab

### Option 1: Upload Manually

Upload your dataset folder to Google Drive, then access it in Colab.

### Option 2: Download from Kaggle

```python
# Install Kaggle API
!pip install -q kaggle

# Upload your kaggle.json credentials
from google.colab import files
files.upload()  # Upload kaggle.json

# Move to correct location
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d <dataset-name>
!unzip <dataset-name>.zip -d ./dataset/
```

### Option 3: Download from URL

```python
import gdown
import zipfile

# Example: Download from Google Drive
file_id = "YOUR_FILE_ID"
url = f"https://drive.google.com/uc?id={file_id}"
output = "dataset.zip"

gdown.download(url, output, quiet=False)

# Extract
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('./dataset/')
```

## 📝 Training Configuration for Colab

Edit these settings in `train.py` for Colab:

```python
# Increase batch size for GPU
BATCH_SIZE = 64  # Default: 32

# More epochs for better accuracy
NUM_EPOCHS = 30  # Default: 20

# Use pretrained model for faster training
USE_PRETRAINED = True  # Default: False
```

## 🔍 Monitoring Training

### View Training Progress

Training will display:
- Loss per batch
- Training accuracy per epoch
- Validation accuracy per epoch
- Best model saved notifications

### Check Model Files

```python
!ls -lh model*.pth
!ls -lh class_names.json
```

### Test Model in Colab

While you can't run the full simulation in Colab (no webcam), you can test the model:

```python
import torch
from model import GestureCNN
from PIL import Image
from torchvision import transforms
import json

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureCNN(num_classes=5)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Test on a sample image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load and predict
img = Image.open('./dataset/forward/image1.jpg')  # Replace with your image
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    gesture = class_names[predicted.item()]
    
print(f"Predicted gesture: {gesture}")
```

## 💾 Saving Progress

To avoid losing work:

1. **Auto-save**: Colab auto-saves to Drive
2. **Manual checkpoints**: Save model periodically during training
3. **Download often**: Download important files to your local machine

## ⚠️ Limitations in Colab

- **No real-time webcam**: Use local Python for gesture detection
- **Session timeout**: Free Colab sessions timeout after 12 hours
- **GPU limits**: Free GPU access is limited (use wisely)
- **Pygame won't work**: Car simulation requires local display

## 🎯 Recommended Workflow

**Best Practice**:
1. **Organize dataset** → Local or Drive
2. **Train model** → Google Colab (GPU)
3. **Download model** → model.pth
4. **Run simulation** → Local Python (webcam + Pygame)

This approach combines the best of both:
- Fast GPU training in Colab
- Full functionality locally

## 📚 Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [Colab GPU Usage Guide](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Mounting Google Drive](https://colab.research.google.com/notebooks/io.ipynb)

---

**Happy Training! 🚀**