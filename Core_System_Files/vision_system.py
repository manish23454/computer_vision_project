"""
Real-Time Computer Vision Tracking System
Main system that integrates landmark detection, face recognition, and Arduino communication
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
import config
from .landmark_detector import LandmarkDetector
from .face_recognizer import FaceRecognizer
from .serial_communicator import ArduinoCommunicator, DataFormatter


class VisionTrackingSystem:
    """Main vision system coordinating all components"""
    
    def __init__(self):
        """Initialize the complete vision tracking system"""
        print("="*60)
        print("Real-Time Computer Vision Tracking System")
        print("="*60)
        
        # Initialize components
        self.detector = LandmarkDetector(mode=config.DETECTION_MODE)
        self.recognizer = FaceRecognizer() if config.ENABLE_FACE_RECOGNITION else None
        self.communicator = ArduinoCommunicator()
        
        # Initialize camera
        self.camera = None
        self.is_running = False
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        print("="*60)
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.camera.isOpened():
            print("✗ Failed to open camera")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.FPS)
        
        if config.DEBUG_MODE:
            print(f"✓ Camera initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} @ {config.FPS}fps")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame through the entire pipeline
        
        Returns:
            annotated_frame: Frame with visual overlays
            data: Dictionary with tracking data
        """
        h, w = frame.shape[:2]
        
        # Step 1: Detect landmarks
        position, distance_ratio, annotated_frame = self.detector.detect(frame)
        
        # Step 2: Calculate position labels
        if position is not None:
            horizontal, vertical = self.detector.calculate_position_labels(position, w, h)
            distance = self.detector.calculate_distance_label(distance_ratio)
            x, y = position
        else:
            horizontal, vertical, distance = 'NONE', 'NONE', 'NONE'
            x, y = 0, 0
        
        # Step 3: Face recognition (if enabled and in face mode)
        match = 'NO'
        matched_name = None
        confidence = 0.0
        
        if self.recognizer and config.ENABLE_FACE_RECOGNITION:
            is_match, matched_name, confidence = self.recognizer.recognize(frame)
            match = 'YES' if is_match else 'NO'
            
            # Add match visualization
            if config.SHOW_INFO_OVERLAY and is_match:
                annotated_frame = self.recognizer.visualize_match(
                    annotated_frame, is_match, matched_name, confidence
                )
        
        # Step 4: Create data packet
        data = DataFormatter.create_tracking_data(
            mode=config.DETECTION_MODE,
            horizontal=horizontal,
            vertical=vertical,
            distance=distance,
            match=match,
            x=x,
            y=y
        )
        
        # Add additional info
        data['matched_name'] = matched_name
        data['confidence'] = confidence
        data['fps'] = self.current_fps
        
        # Step 5: Draw info overlay
        if config.SHOW_INFO_OVERLAY:
            annotated_frame = self._draw_info_overlay(annotated_frame, data)
        
        return annotated_frame, data
    
    def _draw_info_overlay(self, frame: np.ndarray, data: dict) -> np.ndarray:
        """Draw information overlay on frame"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Create semi-transparent background for text
        overlay_height = 180
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), 
                     config.TEXT_BG_COLOR, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Prepare text lines
        lines = [
            f"MODE: {data['mode']}",
            f"POSITION: H={data['horizontal']} | V={data['vertical']}",
            f"DISTANCE: {data['distance']}",
            f"MATCH: {data['match']}",
            f"FPS: {data['fps']:.1f}"
        ]
        
        if data.get('matched_name'):
            lines.append(f"NAME: {data['matched_name']} ({data['confidence']:.2f})")
        
        # Draw text
        y_offset = 25
        for line in lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       config.OVERLAY_FONT_SCALE,
                       config.TEXT_COLOR,
                       config.OVERLAY_THICKNESS)
            y_offset += 28
        
        # Draw status indicators
        status_text = "TRACKING" if data['horizontal'] != 'NONE' else "NO DETECTION"
        status_color = (0, 255, 0) if data['horizontal'] != 'NONE' else (0, 0, 255)
        cv2.putText(frame, status_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed > 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main loop - run the vision tracking system"""
        if not self.initialize_camera():
            return
        
        self.is_running = True
        
        print("\n" + "="*60)
        print("SYSTEM RUNNING")
        print("="*60)
        print("Controls:")
        print("  [q] - Quit")
        print("  [s] - Save current frame")
        print("  [r] - Reset/recalibrate")
        print("  [c] - Capture reference face")
        print("  [Space] - Pause/Resume")
        print("="*60 + "\n")
        
        paused = False
        frame_count = 0
        
        try:
            while self.is_running:
                if not paused:
                    # Capture frame
                    ret, frame = self.camera.read()
                    
                    if not ret:
                        print("✗ Failed to capture frame")
                        break
                    
                    # Flip frame for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    annotated_frame, data = self.process_frame(frame)
                    
                    # Send data to Arduino
                    self.communicator.send_data(data)
                    
                    # Update FPS
                    self._update_fps()
                    
                    # Display frame
                    cv2.imshow('Vision Tracking System', annotated_frame)
                    
                    frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    print("\nShutting down...")
                    break
                
                elif key == ord('s'):
                    # Save frame
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"✓ Frame saved: {filename}")
                
                elif key == ord('r'):
                    # Reset
                    print("✓ System reset")
                
                elif key == ord('c'):
                    # Capture reference face
                    if self.recognizer:
                        name = input("Enter name for reference face: ")
                        if name:
                            success = self.recognizer.add_reference_face(frame, name)
                            if success:
                                print(f"✓ Reference face captured: {name}")
                            else:
                                print("✗ Failed to capture face")
                
                elif key == ord(' '):
                    # Pause/Resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"✓ System {status}")
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        if self.detector:
            self.detector.cleanup()
        
        if self.communicator:
            self.communicator.close()
        
        print("✓ System shutdown complete")


def main():
    """Entry point"""
    try:
        system = VisionTrackingSystem()
        system.run()
    
    except Exception as e:
        print(f"\n✗ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()