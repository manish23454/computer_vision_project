# Real-Time Computer Vision Tracking System

A modular Python-based computer vision system using OpenCV and MediaPipe for real-time human face and body landmark detection. Features multiple detection modes, face recognition, and Arduino integration for robotics applications.

## 🎯 Features

### Detection Modes
- **Head Mode**: Tracks face center using nose tip
- **Eye Mode**: Tracks eye center (average of both eyes)
- **Mouth Mode**: Tracks mouth center
- **Hand Mode**: Tracks hand position (wrist landmark)
- **Ear Mode**: Tracks ear position (average of both ears)

### Real-Time Tracking
- **Horizontal Position**: LEFT / CENTER / RIGHT
- **Vertical Position**: UP / CENTER / DOWN
- **Distance Estimation**: NEAR / MEDIUM / FAR (based on landmark separation)
- **Face Recognition**: YES / NO (matches against reference dataset)

### Arduino Integration
- Structured serial data format for easy parsing
- Real-time motor control commands
- Position and distance data for autonomous tracking
- Face match indicators

## 📋 Requirements

### Python Dependencies
```bash
pip install opencv-python mediapipe numpy pyserial
```

### Hardware (Optional)
- Webcam (built-in or USB)
- Arduino board (for motor control)
- DC motors or servos
- Motor driver (L298N recommended)
- LED and buzzer for indicators

## 🚀 Quick Start

### 1. Basic Setup

```bash
# Clone or download the project files
cd vision_tracking_system

# Install dependencies
pip install opencv-python mediapipe numpy pyserial
```

### 2. Configure the System

Edit `config.py` to customize settings:

```python
# Change detection mode
DETECTION_MODE = 'head'  # Options: 'head', 'eye', 'mouth', 'hand', 'ear'

# Enable/disable features
ENABLE_SERIAL = False  # Set True when Arduino is connected
ENABLE_FACE_RECOGNITION = True

# Adjust sensitivity
HORIZONTAL_CENTER_THRESHOLD = 0.15  # ±15% for CENTER zone
VERTICAL_CENTER_THRESHOLD = 0.15

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Linux/Mac
# SERIAL_PORT = 'COM3'  # Windows
SERIAL_BAUD_RATE = 9600
```

### 3. Create Face Dataset (Optional)

If you want face recognition:

```bash
python dataset_creator.py
```

- Select option 1 to capture reference faces
- Position your face in frame
- Press SPACE to capture
- Enter name when prompted
- Capture multiple angles for better recognition

### 4. Run the System

```bash
python vision_system.py
```

### 5. Controls

While the system is running:
- **Q**: Quit
- **S**: Save current frame
- **R**: Reset/recalibrate
- **C**: Capture reference face
- **Space**: Pause/Resume

## 📊 Data Format

### Serial Output to Arduino

```
<MODE:HEAD,H:LEFT,V:CENTER,D:NEAR,M:YES,X:320,Y:240>
```

**Fields:**
- `MODE`: Current detection mode
- `H`: Horizontal position (LEFT/CENTER/RIGHT)
- `V`: Vertical position (UP/CENTER/DOWN)
- `D`: Distance estimate (NEAR/MEDIUM/FAR)
- `M`: Face match (YES/NO)
- `X`: X-coordinate in pixels
- `Y`: Y-coordinate in pixels

### Parsing Example (Arduino)

```cpp
// Data arrives as: <MODE:HEAD,H:LEFT,V:CENTER,D:NEAR,M:YES>
String horizontal = data.substring(data.indexOf("H:") + 2, data.indexOf(",V:"));
String vertical = data.substring(data.indexOf("V:") + 2, data.indexOf(",D:"));
String distance = data.substring(data.indexOf("D:") + 2, data.indexOf(",M:"));
String match = data.substring(data.indexOf("M:") + 2, data.indexOf(">"));
```

## 🔧 Arduino Integration

### Hardware Setup

```
Arduino Pins:
- Pin 3  → Motor H Left (PWM)
- Pin 5  → Motor H Right (PWM)
- Pin 6  → Motor V Up (PWM)
- Pin 9  → Motor V Down (PWM)
- Pin 13 → LED (Match Indicator)
- Pin 11 → Buzzer (Alerts)
```

### Upload Arduino Code

1. Open `arduino_receiver.ino` in Arduino IDE
2. Connect your Arduino board
3. Select correct board and port
4. Upload the code

### Enable Serial Communication

In `config.py`:

```python
ENABLE_SERIAL = True
SERIAL_PORT = '/dev/ttyUSB0'  # Your Arduino port
```

## 📁 Project Structure

```
vision_tracking_system/
│
├── config.py                 # Configuration settings (MAIN CONFIG FILE)
├── vision_system.py          # Main system entry point
├── landmark_detector.py      # Landmark detection module
├── face_recognizer.py        # Face recognition module
├── serial_communicator.py    # Arduino communication module
├── dataset_creator.py        # Tool to create face dataset
├── arduino_receiver.ino      # Arduino code for receiving data
│
├── face_dataset/             # Reference face images (created automatically)
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
│
└── README.md                 # This file
```

## 🎨 Switching Detection Modes

Simply change the mode in `config.py`:

```python
# Head tracking (face center)
DETECTION_MODE = 'head'

# Eye tracking
DETECTION_MODE = 'eye'

# Mouth tracking
DETECTION_MODE = 'mouth'

# Hand tracking
DETECTION_MODE = 'hand'

# Ear tracking
DETECTION_MODE = 'ear'
```

No other code changes needed!

## ⚙️ Advanced Configuration

### Adjust Position Thresholds

```python
# In config.py
HORIZONTAL_CENTER_THRESHOLD = 0.15  # Wider center zone = 0.20
VERTICAL_CENTER_THRESHOLD = 0.15    # Narrower center zone = 0.10
```

### Adjust Distance Thresholds

```python
# In config.py
DISTANCE_NEAR_THRESHOLD = 0.25  # Lower = stricter "NEAR"
DISTANCE_FAR_THRESHOLD = 0.15   # Higher = stricter "FAR"
```

### Face Recognition Sensitivity

```python
# In config.py
FACE_RECOGNITION_TOLERANCE = 0.6  # Lower = stricter matching (0.0-1.0)
```

## 🎓 Use Cases

### Educational
- Computer vision demonstrations
- Robotics coursework
- AI/ML projects
- Human-computer interaction studies

### Practical Applications
- Autonomous camera tracking
- Smart surveillance systems
- Interactive installations
- Gesture-controlled devices
- Accessibility tools

## 🔍 Troubleshooting

### Camera Not Opening
```python
# Try different camera index in config.py
CAMERA_INDEX = 0  # Try 0, 1, 2, etc.
```

### Serial Port Not Found
```bash
# Linux/Mac: List available ports
ls /dev/tty*

# Windows: Check Device Manager for COM ports
```

### Poor Face Recognition
- Capture more reference images at different angles
- Adjust lighting conditions
- Increase face recognition tolerance
- Ensure face is clearly visible

### Laggy Performance
```python
# In config.py
FRAME_WIDTH = 640   # Reduce to 320
FRAME_HEIGHT = 480  # Reduce to 240
```

## 🛡️ Safety Features

- **Data Timeout**: Motors stop if no data received for 2 seconds
- **Emergency Stop**: Keyboard interrupt support
- **Visual Feedback**: Real-time FPS and status display
- **Structured Data**: Parse errors won't crash the system

## 📝 Example Usage

### Scenario 1: Face-Following Camera

```python
# config.py
DETECTION_MODE = 'head'
ENABLE_SERIAL = True
ENABLE_FACE_RECOGNITION = False
```

Arduino receives position data and adjusts camera servos to keep face centered.

### Scenario 2: Authorized Person Detection

```python
# config.py
DETECTION_MODE = 'head'
ENABLE_FACE_RECOGNITION = True
```

System only triggers actions when recognized person is detected.

### Scenario 3: Hand Gesture Control

```python
# config.py
DETECTION_MODE = 'hand'
ENABLE_SERIAL = True
```

Control robot movement by moving your hand in different directions.

## 🤝 Contributing

Feel free to modify and extend the system:
- Add new detection modes
- Improve face recognition accuracy
- Add new Arduino commands
- Create custom visualization overlays

## 📄 License

This project is open-source and available for educational and personal use.

## 🙏 Acknowledgments

- **OpenCV**: Computer vision library
- **MediaPipe**: Google's ML solutions for landmark detection
- **Arduino**: Open-source electronics platform

## 📧 Support

For issues or questions:
1. Check configuration settings in `config.py`
2. Review troubleshooting section
3. Verify hardware connections
4. Check serial port permissions

---

**Made for college demonstrations and robotics projects** 🎓🤖