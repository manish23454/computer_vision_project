"""
Face Recognition Module
Detects if the current person exists in the reference dataset
"""

import cv2
import numpy as np
import os
from typing import Optional, List, Tuple
import config


class FaceRecognizer:
    """Handles face recognition and dataset matching"""
    
    def __init__(self, dataset_path: str = None):
        """Initialize face recognizer with dataset"""
        self.dataset_path = dataset_path or config.FACE_DATASET_PATH
        self.reference_encodings = []
        self.reference_names = []
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load reference faces
        self._load_dataset()
    
    def _load_dataset(self):
        """Load reference faces from dataset folder"""
        if not os.path.exists(self.dataset_path):
            if config.DEBUG_MODE:
                print(f"⚠ Dataset folder not found: {self.dataset_path}")
                print(f"  Face recognition will always return 'NO'")
            return
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(self.dataset_path) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            if config.DEBUG_MODE:
                print(f"⚠ No images found in dataset: {self.dataset_path}")
            return
        
        for image_file in image_files:
            image_path = os.path.join(self.dataset_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            # Extract face encoding
            encoding = self._get_face_encoding(image)
            
            if encoding is not None:
                self.reference_encodings.append(encoding)
                # Use filename without extension as name
                name = os.path.splitext(image_file)[0]
                self.reference_names.append(name)
                
                if config.DEBUG_MODE:
                    print(f"  ✓ Loaded reference face: {name}")
        
        if config.DEBUG_MODE:
            print(f"✓ Face Recognizer initialized with {len(self.reference_encodings)} reference faces")
    
    def _get_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face encoding from image
        Uses histogram of oriented gradients (HOG) as a simple feature descriptor
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Use first detected face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (128, 128))
        
        # Calculate HOG features
        encoding = self._calculate_hog_features(face_roi)
        
        return encoding
    
    def _calculate_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate HOG (Histogram of Oriented Gradients) features"""
        # Simple histogram-based features for demonstration
        # In production, use face_recognition library or deep learning models
        
        # Calculate gradients
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        
        # Create histogram of gradients
        hist, _ = np.histogram(angle, bins=32, range=(-180, 180), weights=magnitude)
        
        # Normalize
        hist = hist / (np.sum(hist) + 1e-6)
        
        # Also include pixel intensity histogram
        pixel_hist, _ = np.histogram(image, bins=32, range=(0, 256))
        pixel_hist = pixel_hist / (np.sum(pixel_hist) + 1e-6)
        
        # Combine features
        features = np.concatenate([hist, pixel_hist])
        
        return features
    
    def recognize(self, frame: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """
        Check if face in frame matches any reference face
        
        Returns:
            is_match: True if face matches dataset
            matched_name: Name of matched person or None
            confidence: Confidence score (0-1)
        """
        if not config.ENABLE_FACE_RECOGNITION:
            return False, None, 0.0
        
        if len(self.reference_encodings) == 0:
            # No reference faces loaded
            return False, None, 0.0
        
        # Get encoding for current frame
        current_encoding = self._get_face_encoding(frame)
        
        if current_encoding is None:
            # No face detected
            return False, None, 0.0
        
        # Compare with all reference encodings
        best_match_idx = None
        best_distance = float('inf')
        
        for idx, ref_encoding in enumerate(self.reference_encodings):
            # Calculate Euclidean distance
            distance = np.linalg.norm(current_encoding - ref_encoding)
            
            if distance < best_distance:
                best_distance = distance
                best_match_idx = idx
        
        # Convert distance to confidence (inverse relationship)
        # Normalize distance to 0-1 range
        confidence = max(0, 1 - (best_distance / 2.0))
        
        # Check if best match is within tolerance
        is_match = confidence > (1 - config.FACE_RECOGNITION_TOLERANCE)
        matched_name = self.reference_names[best_match_idx] if is_match else None
        
        if config.DEBUG_MODE and is_match:
            print(f"  ✓ Face matched: {matched_name} (confidence: {confidence:.2f})")
        
        return is_match, matched_name, confidence
    
    def get_match_label(self, frame: np.ndarray) -> str:
        """
        Simple YES/NO label for whether face matches dataset
        
        Returns:
            'YES' if match found, 'NO' otherwise
        """
        is_match, _, _ = self.recognize(frame)
        return 'YES' if is_match else 'NO'
    
    def add_reference_face(self, image: np.ndarray, name: str) -> bool:
        """
        Add a new reference face to the dataset
        
        Args:
            image: Face image
            name: Person's name
        
        Returns:
            True if face added successfully
        """
        encoding = self._get_face_encoding(image)
        
        if encoding is None:
            return False
        
        self.reference_encodings.append(encoding)
        self.reference_names.append(name)
        
        # Save image to dataset folder
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        filename = f"{name}.jpg"
        filepath = os.path.join(self.dataset_path, filename)
        cv2.imwrite(filepath, image)
        
        if config.DEBUG_MODE:
            print(f"✓ Added reference face: {name}")
        
        return True
    
    def visualize_match(self, frame: np.ndarray, is_match: bool, 
                       matched_name: Optional[str] = None, 
                       confidence: float = 0.0) -> np.ndarray:
        """Draw match information on frame"""
        annotated = frame.copy()
        
        # Detect face for bounding box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            # Draw bounding box
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{matched_name} ({confidence:.2f})" if is_match else "Unknown"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated