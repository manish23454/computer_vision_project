# System Workflow & Data Flow Diagrams

## 1. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                             │
│              (Keyboard Controls + Visual Feedback)                  │
└───────────────────────────┬────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────┐
│                    VISION SYSTEM CORE                               │
│                   (vision_system.py)                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Main Loop:                                                   │  │
│  │  1. Capture Frame → 2. Process → 3. Display → 4. Transmit   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────┬──────────────────┬──────────────────┬─────────────────┘
             │                  │                  │
    ┌────────▼────────┐ ┌──────▼──────┐ ┌─────────▼──────────┐
    │   LANDMARK      │ │    FACE     │ │     SERIAL         │
    │   DETECTOR      │ │  RECOGNIZER │ │  COMMUNICATOR      │
    │ (MediaPipe)     │ │    (HOG)    │ │   (Arduino)        │
    └────────┬────────┘ └──────┬──────┘ └─────────┬──────────┘
             │                  │                  │
             └──────────────────┴──────────────────┘
                            │
                ┌───────────▼───────────┐
                │     CONFIGURATION     │
                │      (config.py)      │
                └───────────────────────┘
```

## 2. Detailed Processing Pipeline

```
START
  │
  ▼
┌─────────────────┐
│  Camera Input   │  ← USB Webcam / Built-in Camera
│   640x480 RGB   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flip Frame     │  ← Mirror effect for natural interaction
│  (cv2.flip)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│           LANDMARK DETECTION BRANCH                  │
├─────────────────────────────────────────────────────┤
│  Mode Check (config.DETECTION_MODE)                 │
│  ├─ HEAD  → MediaPipe Face Mesh (468 landmarks)     │
│  ├─ EYE   → Face Mesh → Eye landmarks               │
│  ├─ MOUTH → Face Mesh → Mouth landmarks             │
│  ├─ HAND  → MediaPipe Hands (21 landmarks)          │
│  └─ EAR   → Face Mesh → Ear landmarks               │
├─────────────────────────────────────────────────────┤
│  Output: (x, y) position + distance_ratio           │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    POSITION CALCULATION             │
├─────────────────────────────────────┤
│  Normalize:                         │
│    norm_x = x / frame_width         │
│    norm_y = y / frame_height        │
│                                     │
│  Horizontal:                        │
│    if norm_x < 0.35: LEFT           │
│    elif norm_x > 0.65: RIGHT        │
│    else: CENTER                     │
│                                     │
│  Vertical:                          │
│    if norm_y < 0.35: UP             │
│    elif norm_y > 0.65: DOWN         │
│    else: CENTER                     │
├─────────────────────────────────────┤
│  Output: H=LEFT, V=UP               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    DISTANCE ESTIMATION              │
├─────────────────────────────────────┤
│  Calculate landmark separation:     │
│    distance = euclidean(p1, p2)     │
│    ratio = distance / diagonal      │
│                                     │
│  Classify:                          │
│    if ratio > 0.25: NEAR            │
│    elif ratio < 0.15: FAR           │
│    else: MEDIUM                     │
├─────────────────────────────────────┤
│  Output: D=NEAR                     │
└────────┬────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│         FACE RECOGNITION (Optional)                │
├───────────────────────────────────────────────────┤
│  1. Detect face (Haar Cascade)                    │
│  2. Extract ROI → Resize to 128x128               │
│  3. Compute HOG features (64D vector)             │
│  4. Compare with reference dataset                │
│  5. Calculate confidence                          │
│  6. Threshold matching                            │
├───────────────────────────────────────────────────┤
│  Output: M=YES (confidence: 0.85)                 │
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        DATA FORMATTING                   │
├─────────────────────────────────────────┤
│  Create structured packet:              │
│  {                                       │
│    'mode': 'HEAD',                       │
│    'horizontal': 'LEFT',                 │
│    'vertical': 'UP',                     │
│    'distance': 'NEAR',                   │
│    'match': 'YES',                       │
│    'x': 150,                             │
│    'y': 100                              │
│  }                                       │
├─────────────────────────────────────────┤
│  Format as serial string:               │
│  <MODE:HEAD,H:LEFT,V:UP,D:NEAR,M:YES>   │
└────────┬────────────────────────────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐  ┌───────────────┐  ┌─────────────┐
│  VISUALIZATION │  │ SERIAL OUTPUT │  │  CONSOLE    │
│   (OpenCV)     │  │  (Arduino)    │  │  (Debug)    │
└────────────────┘  └───────────────┘  └─────────────┘
         │                  │
         │                  ▼
         │          ┌───────────────────┐
         │          │  Arduino Parser   │
         │          │  ├─ Decode data   │
         │          │  ├─ Update motors │
         │          │  └─ Set LEDs      │
         │          └───────────────────┘
         │
         ▼
┌────────────────────┐
│  Display Window    │
│  ├─ Video feed     │
│  ├─ Overlays       │
│  ├─ Info panel     │
│  └─ FPS counter    │
└────────┬───────────┘
         │
         ▼
    User Interaction
    (Keyboard Input)
         │
         └──────> Loop back to START
```

## 3. Mode Switching Flow

```
User wants to switch from HEAD to HAND mode

1. Stop current system
   └─> Press 'Q' in running application

2. Edit config.py
   └─> Change: DETECTION_MODE = 'hand'

3. Save file

4. Restart system
   └─> Run: python vision_system.py

5. System reads config
   └─> landmark_detector.py reads DETECTION_MODE

6. Initialize appropriate detector
   └─> MediaPipe Hands (instead of Face Mesh)

7. Use hand-specific landmarks
   └─> Wrist position (landmark 0)
   └─> Reference: Wrist to middle finger (0-9)

8. Same output format
   └─> H:LEFT, V:UP, D:NEAR

9. Arduino receives same data structure
   └─> No Arduino code changes needed!

RESULT: System now tracks hand instead of head
        with ZERO code changes (only config)
```

## 4. Data Flow Through Modules

```
┌──────────────────────────────────────────────────────┐
│                   config.py                           │
│  ┌────────────────────────────────────────────────┐  │
│  │ DETECTION_MODE = 'head'                        │  │
│  │ HORIZONTAL_CENTER_THRESHOLD = 0.15             │  │
│  │ ENABLE_SERIAL = True                           │  │
│  │ ...                                            │  │
│  └────────────────────────────────────────────────┘  │
└───────────────┬──────────────────────────────────────┘
                │ (Read by all modules)
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌────────┐  ┌────────┐  ┌────────┐
│Landmark│  │  Face  │  │ Serial │
│Detector│  │Recogn. │  │  Comm  │
└───┬────┘  └───┬────┘  └───┬────┘
    │           │           │
    └───────────┴───────────┘
                │
                ▼
        ┌───────────────┐
        │ Vision System │
        │  (Integrator) │
        └───────────────┘
```

## 5. Arduino Communication Protocol

```
PYTHON SIDE                          ARDUINO SIDE
    │                                     │
    │  Format data packet                │
    │  <MODE:HEAD,H:LEFT,V:UP>           │
    │                                     │
    ├──────────[Serial Port]─────────────>│
    │         (9600 baud)                 │
    │                                     │
    │                                     ├─ Read until '<'
    │                                     ├─ Buffer data
    │                                     ├─ Read until '>'
    │                                     │
    │                                     ├─ Parse fields:
    │                                     │  ├─ Extract H value
    │                                     │  ├─ Extract V value
    │                                     │  └─ Extract D value
    │                                     │
    │                                     ├─ Execute control:
    │                                     │  ├─ H=LEFT → Motor left
    │                                     │  ├─ V=UP → Motor up
    │                                     │  └─ D=NEAR → Slow speed
    │                                     │
    │<─────────[Feedback]─────────────────┤
    │       "OK" / "ERROR"                │
    │                                     │
    └─────> Continue loop                └────> Wait for next data
```

## 6. Face Recognition Workflow

```
INPUT: Video Frame
    │
    ▼
┌─────────────────┐
│ Convert to Gray │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│  Detect Face       │  ← Haar Cascade Classifier
│  (x, y, w, h)      │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Extract Face ROI  │
│  Resize: 128x128   │
└────────┬───────────┘
         │
         ▼
┌─────────────────────────────┐
│  Compute Features           │
│  ├─ Sobel gradients        │
│  ├─ HOG histogram (32)      │
│  └─ Pixel histogram (32)    │
│  Total: 64D feature vector  │
└────────┬────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Compare with Reference Dataset    │
│  For each reference:               │
│    distance = euclidean(feat,ref)  │
│    confidence = 1 - (distance/2)   │
│  Select best match                 │
└────────┬───────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Threshold Decision          │
│  if confidence > 0.4:        │
│    MATCH = YES               │
│  else:                       │
│    MATCH = NO                │
└────────┬────────────────────┘
         │
         ▼
OUTPUT: M=YES + confidence score
```

## 7. System State Machine

```
     ┌──────────┐
     │  INIT    │ ← System startup
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │  IDLE    │ ← Waiting for camera
     └────┬─────┘
          │
          ▼
     ┌──────────┐
┌────│ TRACKING │────┐
│    └────┬─────┘    │
│         │          │
│         ▼          │
│    ┌────────┐     │
│    │DETECTED│     │  (Main loop)
│    └────┬───┘     │
│         │         │
│         ▼         │
│    ┌────────┐    │
│    │SENDING │    │
│    └────┬───┘    │
│         │        │
└─────────┘        │
          │        │
          ▼        │
     ┌──────────┐  │
     │NO DETECT │──┘ (Target lost)
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │  PAUSED  │ ← User pressed Space
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │ SHUTDOWN │ ← User pressed Q
     └──────────┘
```

## 8. Error Handling Flow

```
┌────────────────┐
│ Operation      │
└───────┬────────┘
        │
        ├─────> Try
        │         │
        │         ▼
        │    ┌────────────┐
        │    │  Success   │──> Continue
        │    └────────────┘
        │
        └─────> Catch Error
                  │
                  ├─ Camera Error
                  │    └─> Try reconnect → Retry (3x) → Fail gracefully
                  │
                  ├─ Serial Error
                  │    └─> Log warning → Continue without serial
                  │
                  ├─ Detection Error
                  │    └─> Skip frame → Continue with next
                  │
                  └─ Critical Error
                       └─> Cleanup → Save state → Exit safely
```

## 9. Performance Optimization Points

```
Frame Capture (1-5ms)
    │
    ▼ [Optional: Skip frames if slow]
Preprocessing (1ms)
    │
    ▼ [GPU acceleration possible]
MediaPipe Detection (20-30ms) ← BOTTLENECK
    │
    ▼ [Negligible]
Position Calc (<1ms)
    │
    ▼ [Negligible]
Distance Calc (<1ms)
    │
    ▼ [Optional: Cache results]
Face Recognition (10-20ms) ← SECONDARY BOTTLENECK
    │
    ▼ [Negligible]
Data Format (<1ms)
    │
    ▼ [Async possible]
Serial Send (5-10ms)
    │
    ▼ [Heavy: Frame rendering]
Display (10-20ms)

TOTAL: ~50-85ms per frame (12-20 FPS typical)

OPTIMIZATION STRATEGIES:
1. Reduce frame resolution
2. Skip face recognition on every Nth frame
3. Use GPU for MediaPipe (if available)
4. Async serial communication
5. Reduce display overlay complexity
```

This comprehensive visual documentation shows the complete flow of the system!