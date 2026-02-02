"""
main.py - Integrated Gesture-Controlled Car Simulation

This script integrates:
- Real-time gesture detection from webcam
- Virtual car simulation with Pygame
- Complete workflow from camera to car control
"""

import sys
from realtime_control import GestureDetector
from car_simulation import CarSimulation


def main():
    """
    Main entry point for integrated simulation
    """
    print("="*60)
    print("GESTURE-CONTROLLED VIRTUAL CAR SIMULATION")
    print("="*60)
    print("\nThis simulation combines:")
    print("  1. Real-time hand gesture detection using webcam")
    print("  2. Virtual car control using detected gestures")
    print("\nGestures:")
    print("  - Forward: Move car up")
    print("  - Backward: Move car down")
    print("  - Left: Move car left")
    print("  - Right: Move car right")
    print("  - Stop: Stop car movement")
    print("\nControls:")
    print("  - Press Q to quit")
    print("="*60)
    
    # Check if model exists
    import os
    if not os.path.exists('model.pth'):
        print("\n❌ ERROR: Trained model not found!")
        print("Please train the model first using: python train.py")
        print("\nMake sure you have:")
        print("  1. Organized your dataset in the dataset/ folder")
        print("  2. Run train.py to train the model")
        print("  3. Verify model.pth file is created")
        return
    
    # Initialize gesture detector
    print("\nInitializing gesture detector...")
    try:
        detector = GestureDetector(model_path='model.pth', use_pretrained=False)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize gesture detector: {e}")
        return
    
    # Initialize car simulation
    print("Initializing car simulation...")
    try:
        simulation = CarSimulation(width=800, height=600)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize simulation: {e}")
        return
    
    # Start integrated simulation
    print("\n✓ Starting integrated simulation...")
    print("Show your hand to the camera and make gestures!")
    print("-" * 60)
    
    try:
        # Get gesture generator
        gesture_generator = detector.run(show_video=True)
        
        # Run simulation with gesture control
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