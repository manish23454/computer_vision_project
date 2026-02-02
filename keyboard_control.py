"""
Keyboard-controlled car simulation for testing
This simulates the gesture control using keyboard input
"""

import cv2
import pygame
import sys
from car_simulation import CarSimulation

class KeyboardGestureSimulator:
    """
    Simulates gesture detection using keyboard input
    """
    
    def __init__(self):
        self.current_gesture = "stop"
        self.gesture_confidence = 1.0
        
        # Gesture mapping
        self.key_to_gesture = {
            ord('w'): 'forward',
            ord('W'): 'forward',
            ord('s'): 'backward', 
            ord('S'): 'backward',
            ord('a'): 'left',
            ord('A'): 'left',
            ord('d'): 'right',
            ord('D'): 'right',
            ord(' '): 'stop'  # spacebar
        }
    
    def run(self, show_video=True):
        """
        Run keyboard-controlled gesture simulation
        """
        # Open webcam (optional - just for display)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  Could not open webcam - using black screen")
            cap = None
        
        print("\nKeyboard Gesture Control Active")
        print("Controls:")
        print("  W - Forward")
        print("  S - Backward") 
        print("  A - Left")
        print("  D - Right")
        print("  SPACE - Stop")
        print("  Q - Quit")
        print("-" * 50)
        
        while True:
            # Read frame from webcam (if available)
            if cap:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                else:
                    frame = None
            else:
                # Create black frame
                frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
            
            if frame is not None and show_video:
                # Add text overlay
                cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 0), -1)
                
                # Display current gesture
                text = f"Gesture: {self.current_gesture.upper()}"
                cv2.putText(frame, text, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display confidence
                conf_text = f"Confidence: {self.gesture_confidence:.2f}"
                cv2.putText(frame, conf_text, (20, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display controls
                cv2.putText(frame, "W/A/S/D: Move, SPACE: Stop, Q: Quit", 
                          (20, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Keyboard Gesture Control', frame)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key in self.key_to_gesture:
                self.current_gesture = self.key_to_gesture[key]
                print(f"Gesture: {self.current_gesture}")
            
            # Yield current state
            yield frame, self.current_gesture, self.gesture_confidence
        
        # Cleanup
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\nKeyboard control stopped")

def main():
    """
    Test keyboard-controlled car simulation
    """
    print("="*60)
    print("KEYBOARD-CONTROLLED CAR SIMULATION")
    print("="*60)
    print("\nThis simulation uses keyboard input instead of gestures")
    print("to control the virtual car.")
    print("\nKeyboard Controls:")
    print("  W - Move car forward")
    print("  S - Move car backward")
    print("  A - Move car left")
    print("  D - Move car right")
    print("  SPACE - Stop car")
    print("  Q - Quit")
    print("="*60)
    
    # Initialize keyboard simulator
    print("\nInitializing keyboard control...")
    simulator = KeyboardGestureSimulator()
    
    # Initialize car simulation
    print("Initializing car simulation...")
    try:
        simulation = CarSimulation(width=800, height=600)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize simulation: {e}")
        return
    
    print("✓ Systems initialized")
    print("\n🎮 Starting keyboard-controlled simulation...")
    print("Use W/A/S/D keys to control the car!")
    print("-" * 60)
    
    try:
        # Get keyboard gesture generator
        gesture_generator = simulator.run(show_video=True)
        
        # Run simulation with keyboard control
        simulation.run_with_gestures(gesture_generator)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        print("✓ Simulation completed")

if __name__ == "__main__":
    main()