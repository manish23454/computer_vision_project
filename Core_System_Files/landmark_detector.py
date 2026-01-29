"""
Landmark Detector Module
Handles face, hand, and body landmark detection using MediaPipe
"""
'''
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

import numpy as np
from typing import Tuple, List, Optional, Dict
import config


class LandmarkDetector:
    """Detects and processes landmarks based on selected mode"""
    
def __init__(self, mode: str = None):
    """Initialize detector with specified mode"""
    self.mode = mode or config.DETECTION_MODE

    if self.mode not in config.LANDMARK_CONFIGS:
        raise ValueError(f"Invalid mode: {self.mode}. Choose from {list(config.LANDMARK_CONFIGS.keys())}")

    self.mode_config = config.LANDMARK_CONFIGS[self.mode]

    # Use already imported top-level MediaPipe modules
    self.mp_face_mesh = mp_face_mesh
    self.mp_hands = mp_hands
    self.mp_drawing = mp_drawing

    # Initialize detectors based on mode requirements
    self.face_mesh = None
    self.hands = None

    if self.mode_config['use_face']:
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
        )

    if self.mode_config['use_hands']:
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
        )
        
        if config.DEBUG_MODE:
            print(f"✓ Landmark Detector initialized in '{self.mode}' mode")
            print(f"  Description: {self.mode_config['description']}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray]:
        """
        Detect landmarks and calculate position
        
        Returns:
            position: (x, y) tuple of landmark position or None
            distance_ratio: Estimated distance ratio or None
            annotated_frame: Frame with visual annotations
        """
        if frame is None:
            return None, None, frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        position = None
        distance_ratio = None
        annotated_frame = frame.copy()
        
        # Process based on mode
        if self.mode_config['use_face']:
            position, distance_ratio = self._process_face(rgb_frame, annotated_frame, w, h)
        
        elif self.mode_config['use_hands']:
            position, distance_ratio = self._process_hands(rgb_frame, annotated_frame, w, h)
        
        # Draw center lines
        if config.SHOW_INFO_OVERLAY:
            self._draw_center_lines(annotated_frame, w, h)
        
        return position, distance_ratio, annotated_frame
    
    def _process_face(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process face landmarks"""
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            # Calculate Euclidean distance between reference points
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Normalize by frame diagonal
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw tracking indicator
        if config.SHOW_LANDMARKS:
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0]-15, position[1]), 
                    (position[0]+15, position[1]), (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0], position[1]-15), 
                    (position[0], position[1]+15), (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _process_hands(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process hand landmarks"""
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw hand skeleton
        if config.SHOW_LANDMARKS:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _draw_center_lines(self, frame, w, h):
        """Draw center reference lines"""
        # Vertical center line
        cv2.line(frame, (w//2, 0), (w//2, h), config.CENTER_LINE_COLOR, 1)
        # Horizontal center line
        cv2.line(frame, (0, h//2), (w, h//2), config.CENTER_LINE_COLOR, 1)
        
        # Draw center zone rectangle
        left_threshold = int(w * config.HORIZONTAL_POSITIONS['left_threshold'])
        right_threshold = int(w * config.HORIZONTAL_POSITIONS['right_threshold'])
        up_threshold = int(h * config.VERTICAL_POSITIONS['up_threshold'])
        down_threshold = int(h * config.VERTICAL_POSITIONS['down_threshold'])
        
        cv2.rectangle(frame, 
                     (left_threshold, up_threshold),
                     (right_threshold, down_threshold),
                     config.CENTER_LINE_COLOR, 1)
    
    def calculate_position_labels(self, position: Tuple[int, int], 
                                  frame_width: int, 
                                  frame_height: int) -> Tuple[str, str]:
        """
        Calculate horizontal and vertical position labels
        
        Returns:
            horizontal_label: 'LEFT', 'CENTER', or 'RIGHT'
            vertical_label: 'UP', 'CENTER', or 'DOWN'
        """
        if position is None:
            return 'NONE', 'NONE'
        
        x, y = position
        
        # Normalize to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Horizontal position
        if norm_x < config.HORIZONTAL_POSITIONS['left_threshold']:
            horizontal = 'LEFT'
        elif norm_x > config.HORIZONTAL_POSITIONS['right_threshold']:
            horizontal = 'RIGHT'
        else:
            horizontal = 'CENTER'
        
        # Vertical position
        if norm_y < config.VERTICAL_POSITIONS['up_threshold']:
            vertical = 'UP'
        elif norm_y > config.VERTICAL_POSITIONS['down_threshold']:
            vertical = 'DOWN'
        else:
            vertical = 'CENTER'
        
        return horizontal, vertical
    
    def calculate_distance_label(self, distance_ratio: Optional[float]) -> str:
        """
        Calculate distance label based on landmark separation
        
        Returns:
            'NEAR', 'MEDIUM', or 'FAR'
        """
        if distance_ratio is None:
            return 'UNKNOWN'
        
        if distance_ratio > config.DISTANCE_NEAR_THRESHOLD:
            return 'NEAR'
        elif distance_ratio < config.DISTANCE_FAR_THRESHOLD:
            return 'FAR'
        else:
            return 'MEDIUM'
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands:
            self.hands.close()
        
        if config.DEBUG_MODE:
            print("✓ Landmark Detector cleaned up")

'''













"""
Landmark Detector Module
Handles face, hand, and body landmark detection using MediaPipe
"""
'''
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict
import config

class LandmarkDetector:
    """Detects and processes landmarks based on selected mode"""
    
    def __init__(self, mode: str = None):
        """Initialize detector with specified mode"""
        self.mode = mode or config.DETECTION_MODE

        if self.mode not in config.LANDMARK_CONFIGS:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from {list(config.LANDMARK_CONFIGS.keys())}")

        self.mode_config = config.LANDMARK_CONFIGS[self.mode]

        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize detectors based on mode requirements
        self.face_mesh = None
        self.hands = None

        if self.mode_config['use_face']:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=config.MAX_NUM_FACES,
                refine_landmarks=True,
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
            )

        if self.mode_config['use_hands']:
            self.hands = self.mp_hands.Hands(
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
            )
        
        if config.DEBUG_MODE:
            print(f"✓ Landmark Detector initialized in '{self.mode}' mode")
            print(f"  Description: {self.mode_config['description']}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray]:
        """
        Detect landmarks and calculate position
        
        Returns:
            position: (x, y) tuple of landmark position or None
            distance_ratio: Estimated distance ratio or None
            annotated_frame: Frame with visual annotations
        """
        if frame is None:
            return None, None, frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        position = None
        distance_ratio = None
        annotated_frame = frame.copy()
        
        # Process based on mode
        if self.mode_config['use_face']:
            position, distance_ratio = self._process_face(rgb_frame, annotated_frame, w, h)
        
        elif self.mode_config['use_hands']:
            position, distance_ratio = self._process_hands(rgb_frame, annotated_frame, w, h)
        
        # Draw center lines
        if config.SHOW_INFO_OVERLAY:
            self._draw_center_lines(annotated_frame, w, h)
        
        return position, distance_ratio, annotated_frame
    
    def _process_face(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process face landmarks"""
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            # Calculate Euclidean distance between reference points
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Normalize by frame diagonal
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw tracking indicator
        if config.SHOW_LANDMARKS:
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0]-15, position[1]), 
                    (position[0]+15, position[1]), (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0], position[1]-15), 
                    (position[0], position[1]+15), (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _process_hands(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process hand landmarks"""
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw hand skeleton
        if config.SHOW_LANDMARKS:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _draw_center_lines(self, frame, w, h):
        """Draw center reference lines"""
        # Vertical center line
        cv2.line(frame, (w//2, 0), (w//2, h), config.CENTER_LINE_COLOR, 1)
        # Horizontal center line
        cv2.line(frame, (0, h//2), (w, h//2), config.CENTER_LINE_COLOR, 1)
        
        # Draw center zone rectangle
        left_threshold = int(w * config.HORIZONTAL_POSITIONS['left_threshold'])
        right_threshold = int(w * config.HORIZONTAL_POSITIONS['right_threshold'])
        up_threshold = int(h * config.VERTICAL_POSITIONS['up_threshold'])
        down_threshold = int(h * config.VERTICAL_POSITIONS['down_threshold'])
        
        cv2.rectangle(frame, 
                     (left_threshold, up_threshold),
                     (right_threshold, down_threshold),
                     config.CENTER_LINE_COLOR, 1)
    
    def calculate_position_labels(self, position: Tuple[int, int], 
                                  frame_width: int, 
                                  frame_height: int) -> Tuple[str, str]:
        """
        Calculate horizontal and vertical position labels
        
        Returns:
            horizontal_label: 'LEFT', 'CENTER', or 'RIGHT'
            vertical_label: 'UP', 'CENTER', or 'DOWN'
        """
        if position is None:
            return 'NONE', 'NONE'
        
        x, y = position
        
        # Normalize to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Horizontal position
        if norm_x < config.HORIZONTAL_POSITIONS['left_threshold']:
            horizontal = 'LEFT'
        elif norm_x > config.HORIZONTAL_POSITIONS['right_threshold']:
            horizontal = 'RIGHT'
        else:
            horizontal = 'CENTER'
        
        # Vertical position
        if norm_y < config.VERTICAL_POSITIONS['up_threshold']:
            vertical = 'UP'
        elif norm_y > config.VERTICAL_POSITIONS['down_threshold']:
            vertical = 'DOWN'
        else:
            vertical = 'CENTER'
        
        return horizontal, vertical
    
    def calculate_distance_label(self, distance_ratio: Optional[float]) -> str:
        """
        Calculate distance label based on landmark separation
        
        Returns:
            'NEAR', 'MEDIUM', or 'FAR'
        """
        if distance_ratio is None:
            return 'UNKNOWN'
        
        if distance_ratio > config.DISTANCE_NEAR_THRESHOLD:
            return 'NEAR'
        elif distance_ratio < config.DISTANCE_FAR_THRESHOLD:
            return 'FAR'
        else:
            return 'MEDIUM'
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands:
            self.hands.close()
        
        if config.DEBUG_MODE:
            print("✓ Landmark Detector cleaned up")
            '''
"""
Landmark Detector Module
Handles face, hand, and body landmark detection using MediaPipe
"""



















'''
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict
import config


class LandmarkDetector:
    """Detects and processes landmarks based on selected mode"""
    
    def __init__(self, mode: str = None):
        """Initialize detector with specified mode"""
        self.mode = mode or config.DETECTION_MODE

        if self.mode not in config.LANDMARK_CONFIGS:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from {list(config.LANDMARK_CONFIGS.keys())}")

        self.mode_config = config.LANDMARK_CONFIGS[self.mode]

        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize detectors based on mode requirements
        self.face_mesh = None
        self.hands = None

        if self.mode_config['use_face']:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=config.MAX_NUM_FACES,
                refine_landmarks=True,
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
            )

        if self.mode_config['use_hands']:
            self.hands = self.mp_hands.Hands(
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
            )
        
        if config.DEBUG_MODE:
            print(f"✓ Landmark Detector initialized in '{self.mode}' mode")
            print(f"  Description: {self.mode_config['description']}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray]:
        """
        Detect landmarks and calculate position
        
        Returns:
            position: (x, y) tuple of landmark position or None
            distance_ratio: Estimated distance ratio or None
            annotated_frame: Frame with visual annotations
        """
        if frame is None:
            return None, None, frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        position = None
        distance_ratio = None
        annotated_frame = frame.copy()
        
        # Process based on mode
        if self.mode_config['use_face']:
            position, distance_ratio = self._process_face(rgb_frame, annotated_frame, w, h)
        
        elif self.mode_config['use_hands']:
            position, distance_ratio = self._process_hands(rgb_frame, annotated_frame, w, h)
        
        # Draw center lines
        if config.SHOW_INFO_OVERLAY:
            self._draw_center_lines(annotated_frame, w, h)
        
        return position, distance_ratio, annotated_frame
    
    def _process_face(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process face landmarks"""
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            # Calculate Euclidean distance between reference points
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Normalize by frame diagonal
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw tracking indicator
        if config.SHOW_LANDMARKS:
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0]-15, position[1]), 
                    (position[0]+15, position[1]), (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0], position[1]-15), 
                    (position[0], position[1]+15), (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _process_hands(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process hand landmarks"""
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw hand skeleton
        if config.SHOW_LANDMARKS:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _draw_center_lines(self, frame, w, h):
        """Draw center reference lines"""
        # Vertical center line
        cv2.line(frame, (w//2, 0), (w//2, h), config.CENTER_LINE_COLOR, 1)
        # Horizontal center line
        cv2.line(frame, (0, h//2), (w, h//2), config.CENTER_LINE_COLOR, 1)
        
        # Draw center zone rectangle
        left_threshold = int(w * config.HORIZONTAL_POSITIONS['left_threshold'])
        right_threshold = int(w * config.HORIZONTAL_POSITIONS['right_threshold'])
        up_threshold = int(h * config.VERTICAL_POSITIONS['up_threshold'])
        down_threshold = int(h * config.VERTICAL_POSITIONS['down_threshold'])
        
        cv2.rectangle(frame, 
                     (left_threshold, up_threshold),
                     (right_threshold, down_threshold),
                     config.CENTER_LINE_COLOR, 1)
    
    def calculate_position_labels(self, position: Tuple[int, int], 
                                  frame_width: int, 
                                  frame_height: int) -> Tuple[str, str]:
        """
        Calculate horizontal and vertical position labels
        
        Returns:
            horizontal_label: 'LEFT', 'CENTER', or 'RIGHT'
            vertical_label: 'UP', 'CENTER', or 'DOWN'
        """
        if position is None:
            return 'NONE', 'NONE'
        
        x, y = position
        
        # Normalize to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Horizontal position
        if norm_x < config.HORIZONTAL_POSITIONS['left_threshold']:
            horizontal = 'LEFT'
        elif norm_x > config.HORIZONTAL_POSITIONS['right_threshold']:
            horizontal = 'RIGHT'
        else:
            horizontal = 'CENTER'
        
        # Vertical position
        if norm_y < config.VERTICAL_POSITIONS['up_threshold']:
            vertical = 'UP'
        elif norm_y > config.VERTICAL_POSITIONS['down_threshold']:
            vertical = 'DOWN'
        else:
            vertical = 'CENTER'
        
        return horizontal, vertical
    
    def calculate_distance_label(self, distance_ratio: Optional[float]) -> str:
        """
        Calculate distance label based on landmark separation
        
        Returns:
            'NEAR', 'MEDIUM', or 'FAR'
        """
        if distance_ratio is None:
            return 'UNKNOWN'
        
        if distance_ratio > config.DISTANCE_NEAR_THRESHOLD:
            return 'NEAR'
        elif distance_ratio < config.DISTANCE_FAR_THRESHOLD:
            return 'FAR'
        else:
            return 'MEDIUM'
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands:
            self.hands.close()
        
        if config.DEBUG_MODE:
            print("✓ Landmark Detector cleaned up")
            


















"""
Landmark Detector Module
Handles face, hand, and body landmark detection using MediaPipe
Updated for MediaPipe 0.10.32+
"""
''''''''
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict
import config
import time


class LandmarkDetector:
    """Detects and processes landmarks based on selected mode"""
    
    def __init__(self, mode: str = None):
        """Initialize detector with specified mode"""
        self.mode = mode or config.DETECTION_MODE

        if self.mode not in config.LANDMARK_CONFIGS:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from {list(config.LANDMARK_CONFIGS.keys())}")

        self.mode_config = config.LANDMARK_CONFIGS[self.mode]

        # Initialize MediaPipe modules
        self.mp_face_mesh = None
        self.mp_hands = None
        self.face_mesh = None
        self.hands = None

        if self.mode_config['use_face']:
            # Try to initialize face mesh with new API
            try:
                # For MediaPipe 0.10.32
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=None),  # Will use default model
                    num_faces=config.MAX_NUM_FACES,
                    running_mode=VisionRunningMode.VIDEO,
                    min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                    min_face_presence_confidence=config.FACE_TRACKING_CONFIDENCE
                )
                
                self.face_mesh = FaceLandmarker.create_from_options(options)
                if config.DEBUG_MODE:
                    print("✓ Face landmarker initialized with tasks API")
                    
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"Note: Could not initialize face mesh with tasks API: {e}")
                    print("Falling back to older API style...")
                # Fallback to try older style
                try:
                    self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=config.MAX_NUM_FACES,
                        refine_landmarks=True,
                        min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                        min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
                    )
                except:
                    print("✗ Cannot initialize face mesh with any API")
                    self.face_mesh = None

        if self.mode_config['use_hands']:
            # Try to initialize hands with new API
            try:
                # For MediaPipe 0.10.32
                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=None),  # Will use default model
                    num_hands=config.MAX_NUM_HANDS,
                    running_mode=VisionRunningMode.VIDEO,
                    min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=config.HAND_TRACKING_CONFIDENCE
                )
                
                self.hands = HandLandmarker.create_from_options(options)
                if config.DEBUG_MODE:
                    print("✓ Hand landmarker initialized with tasks API")
                    
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"Note: Could not initialize hands with tasks API: {e}")
                    print("Falling back to older API style...")
                # Fallback to try older style
                try:
                    self.hands = mp.solutions.hands.Hands(
                        max_num_hands=config.MAX_NUM_HANDS,
                        min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                        min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
                    )
                except:
                    print("✗ Cannot initialize hands with any API")
                    self.hands = None
        
        if config.DEBUG_MODE:
            print(f"✓ Landmark Detector initialized in '{self.mode}' mode")
            print(f"  Description: {self.mode_config['description']}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray]:
        """
        Detect landmarks and calculate position
        
        Returns:
            position: (x, y) tuple of landmark position or None
            distance_ratio: Estimated distance ratio or None
            annotated_frame: Frame with visual annotations
        """
        if frame is None:
            return None, None, frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        position = None
        distance_ratio = None
        annotated_frame = frame.copy()
        
        # Process based on mode
        if self.mode_config['use_face']:
            position, distance_ratio = self._process_face(rgb_frame, annotated_frame, w, h)
        
        elif self.mode_config['use_hands']:
            position, distance_ratio = self._process_hands(rgb_frame, annotated_frame, w, h)
        
        # Draw center lines
        if config.SHOW_INFO_OVERLAY:
            self._draw_center_lines(annotated_frame, w, h)
        
        return position, distance_ratio, annotated_frame
    
    def _process_face(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process face landmarks"""
        try:
            # Check if using new API
            if hasattr(self.face_mesh, 'detect_for_video'):
                # New API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                results = self.face_mesh.detect_for_video(mp_image, timestamp_ms)
                
                if not results.face_landmarks:
                    return None, None
                
                # Use first detected face
                face_landmarks = results.face_landmarks[0]
                
                # Get target landmarks
                target_indices = self.mode_config['landmarks']
                reference_indices = self.mode_config['reference_landmarks']
                
                # Calculate average position of target landmarks
                target_points = []
                for idx in target_indices:
                    if idx < len(face_landmarks):
                        landmark = face_landmarks[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        target_points.append((x, y))
                        
                        if config.SHOW_LANDMARKS:
                            cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
                
                if not target_points:
                    return None, None
                
                # Average position
                avg_x = int(np.mean([p[0] for p in target_points]))
                avg_y = int(np.mean([p[1] for p in target_points]))
                position = (avg_x, avg_y)
                
                # Calculate distance ratio using reference landmarks
                ref_points = []
                for idx in reference_indices:
                    if idx < len(face_landmarks):
                        landmark = face_landmarks[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        ref_points.append((x, y))
                
                if len(ref_points) >= 2:
                    p1, p2 = ref_points[0], ref_points[1]
                    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    frame_diagonal = np.sqrt(w**2 + h**2)
                    distance_ratio = distance_pixels / frame_diagonal
                else:
                    distance_ratio = None
                
            else:
                # Old API
                results = self.face_mesh.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    return None, None
                
                # Use first detected face
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get target landmarks
                target_indices = self.mode_config['landmarks']
                reference_indices = self.mode_config['reference_landmarks']
                
                # Calculate average position of target landmarks
                target_points = []
                for idx in target_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    target_points.append((x, y))
                    
                    if config.SHOW_LANDMARKS:
                        cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
                
                # Average position
                avg_x = int(np.mean([p[0] for p in target_points]))
                avg_y = int(np.mean([p[1] for p in target_points]))
                position = (avg_x, avg_y)
                
                # Calculate distance ratio using reference landmarks
                ref_points = []
                for idx in reference_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    ref_points.append((x, y))
                
                if len(ref_points) >= 2:
                    p1, p2 = ref_points[0], ref_points[1]
                    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    frame_diagonal = np.sqrt(w**2 + h**2)
                    distance_ratio = distance_pixels / frame_diagonal
                else:
                    distance_ratio = None
            
            # Draw tracking indicator
            if config.SHOW_LANDMARKS and position:
                cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
                cv2.line(annotated_frame, (position[0]-15, position[1]), 
                        (position[0]+15, position[1]), (0, 0, 255), 2)
                cv2.line(annotated_frame, (position[0], position[1]-15), 
                        (position[0], position[1]+15), (0, 0, 255), 2)
            
            return position, distance_ratio
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error processing face: {e}")
            return None, None
    
    def _process_hands(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process hand landmarks"""
        try:
            # Check if using new API
            if hasattr(self.hands, 'detect_for_video'):
                # New API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                results = self.hands.detect_for_video(mp_image, timestamp_ms)
                
                if not results.hand_landmarks:
                    return None, None
                
                # Use first detected hand
                hand_landmarks = results.hand_landmarks[0]
                
                # Get target landmarks
                target_indices = self.mode_config['landmarks']
                reference_indices = self.mode_config['reference_landmarks']
                
                # Calculate average position of target landmarks
                target_points = []
                for idx in target_indices:
                    if idx < len(hand_landmarks):
                        landmark = hand_landmarks[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        target_points.append((x, y))
                        
                        if config.SHOW_LANDMARKS:
                            cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
                
                if not target_points:
                    return None, None
                
                # Average position
                avg_x = int(np.mean([p[0] for p in target_points]))
                avg_y = int(np.mean([p[1] for p in target_points]))
                position = (avg_x, avg_y)
                
                # Calculate distance ratio using reference landmarks
                ref_points = []
                for idx in reference_indices:
                    if idx < len(hand_landmarks):
                        landmark = hand_landmarks[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        ref_points.append((x, y))
                
                if len(ref_points) >= 2:
                    p1, p2 = ref_points[0], ref_points[1]
                    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    frame_diagonal = np.sqrt(w**2 + h**2)
                    distance_ratio = distance_pixels / frame_diagonal
                else:
                    distance_ratio = None
                
                # Draw hand landmarks manually for new API
                if config.SHOW_LANDMARKS:
                    for landmark in hand_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Draw connections
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
                    ]
                    
                    for start_idx, end_idx in connections:
                        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                            start = hand_landmarks[start_idx]
                            end = hand_landmarks[end_idx]
                            start_pt = (int(start.x * w), int(start.y * h))
                            end_pt = (int(end.x * w), int(end.y * h))
                            cv2.line(annotated_frame, start_pt, end_pt, (0, 255, 0), 2)
                    
                    if position:
                        cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
                
            else:
                # Old API
                results = self.hands.process(rgb_frame)
                
                if not results.multi_hand_landmarks:
                    return None, None
                
                # Use first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get target landmarks
                target_indices = self.mode_config['landmarks']
                reference_indices = self.mode_config['reference_landmarks']
                
                # Calculate average position of target landmarks
                target_points = []
                for idx in target_indices:
                    landmark = hand_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    target_points.append((x, y))
                    
                    if config.SHOW_LANDMARKS:
                        cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
                
                # Average position
                avg_x = int(np.mean([p[0] for p in target_points]))
                avg_y = int(np.mean([p[1] for p in target_points]))
                position = (avg_x, avg_y)
                
                # Calculate distance ratio using reference landmarks
                ref_points = []
                for idx in reference_indices:
                    landmark = hand_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    ref_points.append((x, y))
                
                if len(ref_points) >= 2:
                    p1, p2 = ref_points[0], ref_points[1]
                    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    frame_diagonal = np.sqrt(w**2 + h**2)
                    distance_ratio = distance_pixels / frame_diagonal
                else:
                    distance_ratio = None
                
                # Draw hand skeleton for old API
                if config.SHOW_LANDMARKS:
                    try:
                        # Try to use drawing utils if available
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS
                        )
                    except:
                        # Manual drawing fallback
                        for landmark in hand_landmarks.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    
                    if position:
                        cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
            
            return position, distance_ratio
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error processing hands: {e}")
            return None, None
    
    def _draw_center_lines(self, frame, w, h):
        """Draw center reference lines"""
        # Vertical center line
        cv2.line(frame, (w//2, 0), (w//2, h), config.CENTER_LINE_COLOR, 1)
        # Horizontal center line
        cv2.line(frame, (0, h//2), (w, h//2), config.CENTER_LINE_COLOR, 1)
        
        # Draw center zone rectangle
        left_threshold = int(w * config.HORIZONTAL_POSITIONS['left_threshold'])
        right_threshold = int(w * config.HORIZONTAL_POSITIONS['right_threshold'])
        up_threshold = int(h * config.VERTICAL_POSITIONS['up_threshold'])
        down_threshold = int(h * config.VERTICAL_POSITIONS['down_threshold'])
        
        cv2.rectangle(frame, 
                     (left_threshold, up_threshold),
                     (right_threshold, down_threshold),
                     config.CENTER_LINE_COLOR, 1)
    
    def calculate_position_labels(self, position: Tuple[int, int], 
                                  frame_width: int, 
                                  frame_height: int) -> Tuple[str, str]:
        """
        Calculate horizontal and vertical position labels
        
        Returns:
            horizontal_label: 'LEFT', 'CENTER', or 'RIGHT'
            vertical_label: 'UP', 'CENTER', or 'DOWN'
        """
        if position is None:
            return 'NONE', 'NONE'
        
        x, y = position
        
        # Normalize to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Horizontal position
        if norm_x < config.HORIZONTAL_POSITIONS['left_threshold']:
            horizontal = 'LEFT'
        elif norm_x > config.HORIZONTAL_POSITIONS['right_threshold']:
            horizontal = 'RIGHT'
        else:
            horizontal = 'CENTER'
        
        # Vertical position
        if norm_y < config.VERTICAL_POSITIONS['up_threshold']:
            vertical = 'UP'
        elif norm_y > config.VERTICAL_POSITIONS['down_threshold']:
            vertical = 'DOWN'
        else:
            vertical = 'CENTER'
        
        return horizontal, vertical
    
    def calculate_distance_label(self, distance_ratio: Optional[float]) -> str:
        """
        Calculate distance label based on landmark separation
        
        Returns:
            'NEAR', 'MEDIUM', or 'FAR'
        """
        if distance_ratio is None:
            return 'UNKNOWN'
        
        if distance_ratio > config.DISTANCE_NEAR_THRESHOLD:
            return 'NEAR'
        elif distance_ratio < config.DISTANCE_FAR_THRESHOLD:
            return 'FAR'
        else:
            return 'MEDIUM'
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except:
                pass
        if self.hands:
            try:
                self.hands.close()
            except:
                pass
        
        if config.DEBUG_MODE:
            print("✓ Landmark Detector cleaned up")




'''













"""
Landmark Detector Module
Handles face, hand, and body landmark detection using MediaPipe
Compatible with MediaPipe 0.10.11
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict
import config


class LandmarkDetector:
    """Detects and processes landmarks based on selected mode"""
    
    def __init__(self, mode: str = None):
        """Initialize detector with specified mode"""
        self.mode = mode or config.DETECTION_MODE

        if self.mode not in config.LANDMARK_CONFIGS:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from {list(config.LANDMARK_CONFIGS.keys())}")

        self.mode_config = config.LANDMARK_CONFIGS[self.mode]

        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize detectors based on mode requirements
        self.face_mesh = None
        self.hands = None

        if self.mode_config['use_face']:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=config.MAX_NUM_FACES,
                refine_landmarks=True,
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
            )

        if self.mode_config['use_hands']:
            self.hands = self.mp_hands.Hands(
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
            )
        
        if config.DEBUG_MODE:
            print(f"✓ Landmark Detector initialized in '{self.mode}' mode")
            print(f"  Description: {self.mode_config['description']}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray]:
        """
        Detect landmarks and calculate position
        
        Returns:
            position: (x, y) tuple of landmark position or None
            distance_ratio: Estimated distance ratio or None
            annotated_frame: Frame with visual annotations
        """
        if frame is None:
            return None, None, frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        position = None
        distance_ratio = None
        annotated_frame = frame.copy()
        
        # Process based on mode
        if self.mode_config['use_face']:
            position, distance_ratio = self._process_face(rgb_frame, annotated_frame, w, h)
        
        elif self.mode_config['use_hands']:
            position, distance_ratio = self._process_hands(rgb_frame, annotated_frame, w, h)
        
        # Draw center lines
        if config.SHOW_INFO_OVERLAY:
            self._draw_center_lines(annotated_frame, w, h)
        
        return position, distance_ratio, annotated_frame
    
    def _process_face(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process face landmarks"""
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            # Calculate Euclidean distance between reference points
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Normalize by frame diagonal
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw tracking indicator
        if config.SHOW_LANDMARKS:
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0]-15, position[1]), 
                    (position[0]+15, position[1]), (0, 0, 255), 2)
            cv2.line(annotated_frame, (position[0], position[1]-15), 
                    (position[0], position[1]+15), (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _process_hands(self, rgb_frame, annotated_frame, w, h) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Process hand landmarks"""
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get target landmarks
        target_indices = self.mode_config['landmarks']
        reference_indices = self.mode_config['reference_landmarks']
        
        # Calculate average position of target landmarks
        target_points = []
        for idx in target_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append((x, y))
            
            if config.SHOW_LANDMARKS:
                cv2.circle(annotated_frame, (x, y), 5, config.LANDMARK_COLOR, -1)
        
        # Average position
        avg_x = int(np.mean([p[0] for p in target_points]))
        avg_y = int(np.mean([p[1] for p in target_points]))
        position = (avg_x, avg_y)
        
        # Calculate distance ratio using reference landmarks
        ref_points = []
        for idx in reference_indices:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            ref_points.append((x, y))
        
        if len(ref_points) >= 2:
            p1, p2 = ref_points[0], ref_points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            frame_diagonal = np.sqrt(w**2 + h**2)
            distance_ratio = distance_pixels / frame_diagonal
        else:
            distance_ratio = None
        
        # Draw hand skeleton
        if config.SHOW_LANDMARKS:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            cv2.circle(annotated_frame, position, 10, (0, 0, 255), 2)
        
        return position, distance_ratio
    
    def _draw_center_lines(self, frame, w, h):
        """Draw center reference lines"""
        # Vertical center line
        cv2.line(frame, (w//2, 0), (w//2, h), config.CENTER_LINE_COLOR, 1)
        # Horizontal center line
        cv2.line(frame, (0, h//2), (w, h//2), config.CENTER_LINE_COLOR, 1)
        
        # Draw center zone rectangle
        left_threshold = int(w * config.HORIZONTAL_POSITIONS['left_threshold'])
        right_threshold = int(w * config.HORIZONTAL_POSITIONS['right_threshold'])
        up_threshold = int(h * config.VERTICAL_POSITIONS['up_threshold'])
        down_threshold = int(h * config.VERTICAL_POSITIONS['down_threshold'])
        
        cv2.rectangle(frame, 
                     (left_threshold, up_threshold),
                     (right_threshold, down_threshold),
                     config.CENTER_LINE_COLOR, 1)
    
    def calculate_position_labels(self, position: Tuple[int, int], 
                                  frame_width: int, 
                                  frame_height: int) -> Tuple[str, str]:
        """
        Calculate horizontal and vertical position labels
        
        Returns:
            horizontal_label: 'LEFT', 'CENTER', or 'RIGHT'
            vertical_label: 'UP', 'CENTER', or 'DOWN'
        """
        if position is None:
            return 'NONE', 'NONE'
        
        x, y = position
        
        # Normalize to 0-1 range
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Horizontal position
        if norm_x < config.HORIZONTAL_POSITIONS['left_threshold']:
            horizontal = 'LEFT'
        elif norm_x > config.HORIZONTAL_POSITIONS['right_threshold']:
            horizontal = 'RIGHT'
        else:
            horizontal = 'CENTER'
        
        # Vertical position
        if norm_y < config.VERTICAL_POSITIONS['up_threshold']:
            vertical = 'UP'
        elif norm_y > config.VERTICAL_POSITIONS['down_threshold']:
            vertical = 'DOWN'
        else:
            vertical = 'CENTER'
        
        return horizontal, vertical
    
    def calculate_distance_label(self, distance_ratio: Optional[float]) -> str:
        """
        Calculate distance label based on landmark separation
        
        Returns:
            'NEAR', 'MEDIUM', or 'FAR'
        """
        if distance_ratio is None:
            return 'UNKNOWN'
        
        if distance_ratio > config.DISTANCE_NEAR_THRESHOLD:
            return 'NEAR'
        elif distance_ratio < config.DISTANCE_FAR_THRESHOLD:
            return 'FAR'
        else:
            return 'MEDIUM'
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands:
            self.hands.close()
        
        if config.DEBUG_MODE:
            print("✓ Landmark Detector cleaned up")
