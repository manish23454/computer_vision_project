"""
realtime_control.py - Real-time Gesture Detection

This script:
- Captures webcam frames using OpenCV
- Detects hands using MediaPipe
- Crops and preprocesses hand region
- Predicts gesture using trained model
- Maps gestures to car control commands
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
import mediapipe as mp
import json
import os
from model import GestureCNN, get_pretrained_model


class GestureDetector:
    """
    Real-time gesture detection using webcam and trained model
    """
    
    def __init__(self, model_path='model.pth', use_pretrained=False):
        """
        Initialize gesture detector
        
        Args:
            model_path (str): Path to trained model weights
            use_pretrained (bool): Whether model is based on ResNet18
        """
        # ==================== DEVICE SETUP ====================
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # ==================== LOAD CLASS NAMES ====================
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)
        else:
            # Default class names
            self.class_names = ['backward', 'forward', 'left', 'right', 'stop']
        
        print(f"Classes: {self.class_names}")
        
        # ==================== LOAD MODEL ====================
        print("Loading model...")
        num_classes = len(self.class_names)
        
        if use_pretrained:
            self.model = get_pretrained_model(num_classes=num_classes, pretrained=False)
        else:
            self.model = GestureCNN(num_classes=num_classes)
        
        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            print("Please train the model first using train.py")
            exit(1)
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # ==================== IMAGE PREPROCESSING ====================
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ==================== MEDIAPIPE HAND DETECTION ====================
        # Simplified approach - disable MediaPipe for now and use keyboard simulation
        self.hands = None
        self.use_mediapipe = False
        print("⚠️  MediaPipe hand detection disabled - using keyboard simulation")
        print("   Press keys: W(forward), S(backward), A(left), D(right), SPACE(stop)")
        
        # ==================== GESTURE TRACKING ====================
        self.current_gesture = "stop"
        self.gesture_confidence = 0.0
        self.gesture_history = []  # For smoothing predictions
        self.history_size = 5  # Number of frames to average
    
    def detect_hand_region(self, frame):
        """
        Simplified version - return the whole frame as hand region for testing
        """
        if self.hands is None:
            # No MediaPipe - return center region of frame
            h, w, c = frame.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 3
            
            x_min = max(0, center_x - size // 2)
            y_min = max(0, center_y - size // 2)
            x_max = min(w, center_x + size // 2)
            y_max = min(h, center_y + size // 2)
            
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            # Draw a simple rectangle to show "detected" region
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Simulated Hand Region", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return hand_region, (x_min, y_min, x_max - x_min, y_max - y_min), True
        
        # Original MediaPipe code would go here
        return None, None, False
    
    def predict_gesture(self, hand_region):
        """
        Predict gesture from hand region
        
        Args:
            hand_region: Cropped hand image
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        if hand_region is None or hand_region.size == 0:
            return "stop", 0.0
        
        # Preprocess image
        try:
            # Convert BGR to RGB
            hand_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.transform(hand_rgb)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                gesture_idx = predicted.item()
                gesture_conf = confidence.item()
                gesture_name = self.class_names[gesture_idx]
                
                return gesture_name, gesture_conf
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "stop", 0.0
    
    def smooth_gesture(self, gesture, confidence):
        """
        Smooth gesture predictions using history
        
        Args:
            gesture (str): Current predicted gesture
            confidence (float): Prediction confidence
            
        Returns:
            str: Smoothed gesture
        """
        # Add to history
        self.gesture_history.append((gesture, confidence))
        
        # Keep only recent history
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Count occurrences of each gesture
        gesture_counts = {}
        for gest, conf in self.gesture_history:
            if gest not in gesture_counts:
                gesture_counts[gest] = 0
            gesture_counts[gest] += conf  # Weight by confidence
        
        # Return most common gesture
        if gesture_counts:
            smoothed_gesture = max(gesture_counts, key=gesture_counts.get)
            return smoothed_gesture
        
        return gesture
    
    def run(self, show_video=True):
        """
        Run real-time gesture detection
        
        Args:
            show_video (bool): Whether to display video window
            
        Returns:
            generator: Yields (frame, gesture, confidence) tuples
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        print("\nStarting gesture detection...")
        print("Press 'q' to quit")
        print("Gestures: forward, backward, left, right, stop")
        print("-" * 50)
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand region
            hand_region, bbox, has_hand = self.detect_hand_region(frame)
            
            # Predict gesture if hand detected
            if has_hand and hand_region is not None:
                gesture, confidence = self.predict_gesture(hand_region)
                
                # Smooth gesture prediction
                gesture = self.smooth_gesture(gesture, confidence)
                
                # Update current gesture if confidence is high
                if confidence > 0.6:
                    self.current_gesture = gesture
                    self.gesture_confidence = confidence
            else:
                # No hand detected, default to stop
                gesture = "stop"
                confidence = 1.0
                self.current_gesture = gesture
                self.gesture_confidence = confidence
            
            # Display information on frame
            if show_video:
                # Add text background
                cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
                
                # Display gesture
                text = f"Gesture: {self.current_gesture.upper()}"
                cv2.putText(frame, text, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display confidence
                conf_text = f"Confidence: {self.gesture_confidence:.2f}"
                cv2.putText(frame, conf_text, (20, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Gesture Detection', frame)
            
            # Yield current state
            yield frame, self.current_gesture, self.gesture_confidence
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nGesture detection stopped")


def main():
    """
    Test gesture detection independently
    """
    # Initialize detector
    detector = GestureDetector(model_path='model.pth', use_pretrained=False)
    
    # Run detection
    for frame, gesture, confidence in detector.run(show_video=True):
        # Print gesture to console
        print(f"Detected: {gesture} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()