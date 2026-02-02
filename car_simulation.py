"""
car_simulation.py - Virtual Car Simulation with Pygame

This script creates a Pygame simulation where:
- A virtual car appears randomly on the screen
- The car is controlled by gesture commands
- Car movement, boundaries, and scoring are handled
"""

import pygame
import random
import sys
import time

# Initialize pygame mixer to avoid potential issues
pygame.mixer.pre_init()


class Car:
    """
    Virtual car object with position, velocity, and rendering
    """
    
    def __init__(self, x, y, width=60, height=100):
        """
        Initialize car
        
        Args:
            x (int): Initial x position
            y (int): Initial y position
            width (int): Car width
            height (int): Car height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Movement properties
        self.speed = 5  # Pixels per frame
        self.rotation_angle = 0  # Current rotation angle
        
        # Velocity
        self.vx = 0
        self.vy = 0
        
        # Current command
        self.current_command = "stop"
        
        # Create car surface
        self.original_surface = self.create_car_surface()
        self.surface = self.original_surface
    
    def create_car_surface(self):
        """
        Create car sprite
        
        Returns:
            pygame.Surface: Car surface
        """
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Car body (main rectangle)
        pygame.draw.rect(surface, (255, 0, 0), (10, 20, 40, 60))
        
        # Car windows
        pygame.draw.rect(surface, (100, 200, 255), (15, 25, 30, 20))
        pygame.draw.rect(surface, (100, 200, 255), (15, 50, 30, 20))
        
        # Car wheels
        pygame.draw.circle(surface, (50, 50, 50), (10, 30), 8)
        pygame.draw.circle(surface, (50, 50, 50), (50, 30), 8)
        pygame.draw.circle(surface, (50, 50, 50), (10, 70), 8)
        pygame.draw.circle(surface, (50, 50, 50), (50, 70), 8)
        
        # Headlights
        pygame.draw.circle(surface, (255, 255, 0), (20, 15), 5)
        pygame.draw.circle(surface, (255, 255, 0), (40, 15), 5)
        
        return surface
    
    def update_command(self, command):
        """
        Update car movement based on gesture command
        
        Args:
            command (str): Gesture command (forward, backward, left, right, stop)
        """
        self.current_command = command
        
        # Reset velocity
        self.vx = 0
        self.vy = 0
        
        # Set velocity based on command
        if command == "forward":
            self.vy = -self.speed  # Move up (negative y)
            self.rotation_angle = 0
        elif command == "backward":
            self.vy = self.speed  # Move down (positive y)
            self.rotation_angle = 180
        elif command == "left":
            self.vx = -self.speed  # Move left (negative x)
            self.rotation_angle = -90
        elif command == "right":
            self.vx = self.speed  # Move right (positive x)
            self.rotation_angle = 90
        elif command == "stop":
            self.vx = 0
            self.vy = 0
        
        # Rotate car sprite
        self.surface = pygame.transform.rotate(self.original_surface, self.rotation_angle)
    
    def update_position(self, screen_width, screen_height):
        """
        Update car position with boundary checking
        
        Args:
            screen_width (int): Screen width
            screen_height (int): Screen height
        """
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Boundary checking - keep car within screen
        if self.x < 0:
            self.x = 0
        elif self.x + self.width > screen_width:
            self.x = screen_width - self.width
        
        if self.y < 0:
            self.y = 0
        elif self.y + self.height > screen_height:
            self.y = screen_height - self.height
    
    def draw(self, screen):
        """
        Draw car on screen
        
        Args:
            screen: Pygame screen surface
        """
        # Get rotated rect
        rect = self.surface.get_rect(center=(self.x + self.width//2, self.y + self.height//2))
        screen.blit(self.surface, rect.topleft)
        
        # Draw bounding box (for debugging)
        # pygame.draw.rect(screen, (0, 255, 0), (self.x, self.y, self.width, self.height), 2)


class CarSimulation:
    """
    Main car simulation with Pygame
    """
    
    def __init__(self, width=800, height=600):
        """
        Initialize simulation
        
        Args:
            width (int): Window width
            height (int): Window height
        """
        # Initialize Pygame
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gesture-Controlled Virtual Car")
        
        # Clock for frame rate
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Car setup - random initial position
        car_x = random.randint(50, self.width - 150)
        car_y = random.randint(50, self.height - 150)
        self.car = Car(car_x, car_y)
        
        # Colors
        self.bg_color = (50, 50, 50)
        self.text_color = (255, 255, 255)
        self.road_color = (80, 80, 80)
        
        # Font
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Statistics
        self.distance_traveled = 0
        self.start_time = time.time()
        
        # Running flag
        self.running = True
    
    def handle_events(self):
        """
        Handle Pygame events
        
        Returns:
            bool: Whether simulation should continue
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def update_car(self, gesture_command):
        """
        Update car based on gesture command
        
        Args:
            gesture_command (str): Current gesture command
        """
        # Update car command
        self.car.update_command(gesture_command)
        
        # Update car position
        old_x, old_y = self.car.x, self.car.y
        self.car.update_position(self.width, self.height)
        
        # Calculate distance traveled
        dx = self.car.x - old_x
        dy = self.car.y - old_y
        distance = (dx**2 + dy**2)**0.5
        self.distance_traveled += distance
    
    def draw_road(self):
        """
        Draw road/track on background
        """
        # Draw road lines
        line_width = 5
        dash_length = 30
        dash_gap = 20
        
        # Vertical center line
        center_x = self.width // 2
        for y in range(0, self.height, dash_length + dash_gap):
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (center_x, y), (center_x, y + dash_length), line_width)
        
        # Horizontal center line
        center_y = self.height // 2
        for x in range(0, self.width, dash_length + dash_gap):
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (x, center_y), (x + dash_length, center_y), line_width)
        
        # Boundary lines
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (0, 0, self.width, self.height), 3)
    
    def draw_ui(self, gesture_command, gesture_confidence):
        """
        Draw UI elements (gesture info, stats)
        
        Args:
            gesture_command (str): Current gesture
            gesture_confidence (float): Gesture confidence
        """
        # Draw semi-transparent background for UI
        ui_surface = pygame.Surface((self.width, 120), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 180))
        self.screen.blit(ui_surface, (0, 0))
        
        # Gesture command
        gesture_text = self.font_large.render(
            f"Gesture: {gesture_command.upper()}", True, self.text_color)
        self.screen.blit(gesture_text, (20, 20))
        
        # Confidence
        conf_text = self.font_small.render(
            f"Confidence: {gesture_confidence:.2%}", True, self.text_color)
        self.screen.blit(conf_text, (20, 70))
        
        # Distance traveled
        distance_text = self.font_small.render(
            f"Distance: {int(self.distance_traveled)} pixels", True, self.text_color)
        self.screen.blit(distance_text, (self.width - 300, 20))
        
        # Time elapsed
        elapsed_time = time.time() - self.start_time
        time_text = self.font_small.render(
            f"Time: {int(elapsed_time)}s", True, self.text_color)
        self.screen.blit(time_text, (self.width - 300, 50))
        
        # Instructions
        instruction_text = self.font_small.render(
            "Press Q or ESC to quit", True, (200, 200, 200))
        self.screen.blit(instruction_text, (20, self.height - 30))
    
    def render(self, gesture_command="stop", gesture_confidence=0.0):
        """
        Render the simulation frame
        
        Args:
            gesture_command (str): Current gesture command
            gesture_confidence (float): Confidence of gesture prediction
        """
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Draw road
        self.draw_road()
        
        # Draw car
        self.car.draw(self.screen)
        
        # Draw UI
        self.draw_ui(gesture_command, gesture_confidence)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(self.fps)
    
    def run_standalone(self):
        """
        Run simulation in standalone mode with keyboard controls
        (for testing without gesture detection)
        """
        print("\nRunning car simulation in standalone mode")
        print("Controls:")
        print("  W - Forward")
        print("  S - Backward")
        print("  A - Left")
        print("  D - Right")
        print("  SPACE - Stop")
        print("  Q/ESC - Quit")
        print("-" * 50)
        
        gesture_command = "stop"
        
        while self.running:
            # Handle events
            self.running = self.handle_events()
            
            # Keyboard controls
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                gesture_command = "forward"
            elif keys[pygame.K_s]:
                gesture_command = "backward"
            elif keys[pygame.K_a]:
                gesture_command = "left"
            elif keys[pygame.K_d]:
                gesture_command = "right"
            elif keys[pygame.K_SPACE]:
                gesture_command = "stop"
            
            # Update car
            self.update_car(gesture_command)
            
            # Render
            self.render(gesture_command, 1.0)
        
        # Cleanup
        pygame.quit()
        print("\nSimulation ended")
        print(f"Total distance traveled: {int(self.distance_traveled)} pixels")
    
    def run_with_gestures(self, gesture_generator):
        """
        Run simulation with gesture detection
        
        Args:
            gesture_generator: Generator yielding (frame, gesture, confidence)
        """
        print("\nRunning car simulation with gesture control")
        print("Show gestures to the camera to control the car")
        print("Press Q to quit")
        print("-" * 50)
        
        for frame, gesture_command, gesture_confidence in gesture_generator:
            # Handle events
            self.running = self.handle_events()
            if not self.running:
                break
            
            # Update car
            self.update_car(gesture_command)
            
            # Render
            self.render(gesture_command, gesture_confidence)
        
        # Cleanup
        pygame.quit()
        print("\nSimulation ended")
        print(f"Total distance traveled: {int(self.distance_traveled)} pixels")


def main():
    """
    Test simulation in standalone mode
    """
    simulation = CarSimulation(width=800, height=600)
    simulation.run_standalone()


if __name__ == "__main__":
    main()