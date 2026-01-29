"""
Configuration file for Real-Time Computer Vision Tracking System
Easily change detection modes and system parameters here
"""

# ============================================================================
# DETECTION MODE SELECTION
# ============================================================================
# Available modes: 'head', 'eye', 'mouth', 'hand', 'ear'
DETECTION_MODE = 'ear'  # <-- CHANGE THIS TO SWITCH MODES

# ============================================================================
# LANDMARK MAPPINGS FOR EACH MODE
# ============================================================================
# MediaPipe Face Mesh has 468 landmarks
# MediaPipe Hands has 21 landmarks per hand
# MediaPipe Pose has 33 landmarks

LANDMARK_CONFIGS = {
    'head': {
        'landmarks': [1],  # Nose tip (face center)
        'reference_landmarks': [10, 152],  # Forehead to chin for distance
        'description': 'Face center tracking using nose tip',
        'use_face': True,
        'use_hands': False,
        'use_pose': False
    },
    'eye': {
        'landmarks': [468, 473],  # Left eye center, Right eye center
        'reference_landmarks': [33, 263],  # Eye corners for distance
        'description': 'Eye tracking (averages both eyes)',
        'use_face': True,
        'use_hands': False,
        'use_pose': False
    },
    'mouth': {
        'landmarks': [13, 14],  # Upper and lower lips
        'reference_landmarks': [61, 291],  # Mouth corners for distance
        'description': 'Mouth center tracking',
        'use_face': True,
        'use_hands': False,
        'use_pose': False
    },
    'hand': {
        'landmarks': [0],  # Wrist landmark
        'reference_landmarks': [0, 9],  # Wrist to middle finger base for distance
        'description': 'Hand tracking (wrist position)',
        'use_face': False,
        'use_hands': True,
        'use_pose': False
    },
    'ear': {
        'landmarks': [234, 454],  # Left ear, Right ear (approximate)
        'reference_landmarks': [127, 356],  # Ear references for distance
        'description': 'Ear tracking (averages both ears)',
        'use_face': True,
        'use_hands': False,
        'use_pose': False
    }
}

# ============================================================================
# POSITION THRESHOLDS (adjustable for sensitivity)
# ============================================================================
# Horizontal thresholds (fraction of frame width)
HORIZONTAL_CENTER_THRESHOLD = 0.3  # ±15% is considered CENTER
HORIZONTAL_POSITIONS = {
    'left_threshold': 0.5 - HORIZONTAL_CENTER_THRESHOLD,
    'right_threshold': 0.5 + HORIZONTAL_CENTER_THRESHOLD
}

# Vertical thresholds (fraction of frame height)
VERTICAL_CENTER_THRESHOLD = 0.15  # ±15% is considered CENTER
VERTICAL_POSITIONS = {
    'up_threshold': 0.5 - VERTICAL_CENTER_THRESHOLD,
    'down_threshold': 0.5 + VERTICAL_CENTER_THRESHOLD
}

# Distance thresholds (based on landmark separation ratio)
DISTANCE_NEAR_THRESHOLD = 0.5  # Above this ratio = NEAR
DISTANCE_FAR_THRESHOLD = 0.5  # Below this ratio = FAR
# Between these values = MEDIUM

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_INDEX = 0  # Default webcam (change if using external camera)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10

# ============================================================================
# SERIAL COMMUNICATION SETTINGS
# ============================================================================
SERIAL_PORT = '/dev/ttyUSB0'  # Linux/Mac: /dev/ttyUSB0 or /dev/ttyACM0
                               # Windows: 'COM3', 'COM4', etc.
SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 1
ENABLE_SERIAL = False  # Set to True when Arduino is connected

# ============================================================================
# FACE RECOGNITION SETTINGS
# ============================================================================
FACE_DATASET_PATH = 'face_dataset'  # Folder containing reference images
FACE_RECOGNITION_TOLERANCE = 0.1    # Lower = stricter matching (0.0-1.0)
ENABLE_FACE_RECOGNITION = True      # Set to False to disable face matching

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
SHOW_LANDMARKS = True      # Draw landmarks on video feed
SHOW_INFO_OVERLAY = True   # Show position/distance info on screen
OVERLAY_FONT_SCALE = 0.6
OVERLAY_THICKNESS = 2
LANDMARK_COLOR = (0, 255, 0)      # Green (BGR)
CENTER_LINE_COLOR = (255, 0, 0)   # Blue (BGR)
TEXT_COLOR = (255, 255, 255)      # White (BGR)
TEXT_BG_COLOR = (0, 0, 0)         # Black (BGR)

# ============================================================================
# MEDIAPIPE SETTINGS
# ============================================================================
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2
MAX_NUM_FACES = 1

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
DEBUG_MODE = True  # Print debug information to console
SAVE_DEBUG_FRAMES = False  # Save frames for debugging
DEBUG_FRAME_PATH = 'debug_frames'