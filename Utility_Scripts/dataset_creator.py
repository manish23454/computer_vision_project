"""
Serial Communication Module
Handles data transmission to Arduino in structured format
"""

import serial
import time
from typing import Optional, Dict, Any
import config


class ArduinoCommunicator:
    """Manages serial communication with Arduino"""
    
    def __init__(self, port: str = None, baud_rate: int = None):
        """Initialize serial connection"""
        self.port = port or config.SERIAL_PORT
        self.baud_rate = baud_rate or config.SERIAL_BAUD_RATE
        self.serial_connection = None
        self.is_connected = False
        
        if config.ENABLE_SERIAL:
            self._connect()
        else:
            if config.DEBUG_MODE:
                print("⚠ Serial communication disabled in config")
                print("  Data will be printed to console only")
    
    def _connect(self):
        """Establish serial connection"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=config.SERIAL_TIMEOUT,
                write_timeout=2
            )
            
            # Wait for Arduino to reset
            time.sleep(2)
            
            self.is_connected = True
            
            if config.DEBUG_MODE:
                print(f"✓ Serial connection established: {self.port} @ {self.baud_rate} baud")
        
        except serial.SerialException as e:
            self.is_connected = False
            if config.DEBUG_MODE:
                print(f"✗ Failed to connect to {self.port}: {e}")
                print("  Continuing without serial connection...")
    
    def send_data(self, data_dict: Dict[str, Any]) -> bool:
        """
        Send structured data to Arduino
        
        Data format: <MODE:value,H:value,V:value,D:value,M:value>
        Example: <MODE:head,H:LEFT,V:CENTER,D:NEAR,M:YES>
        
        Args:
            data_dict: Dictionary with keys: mode, horizontal, vertical, distance, match
        
        Returns:
            True if sent successfully
        """
        # Format data string
        data_string = self._format_data(data_dict)
        
        # Print to console (always)
        if config.DEBUG_MODE:
            print(f"→ {data_string}")
        
        # Send via serial if connected
        if self.is_connected and self.serial_connection:
            try:
                self.serial_connection.write(data_string.encode('utf-8'))
                self.serial_connection.flush()
                return True
            
            except serial.SerialException as e:
                if config.DEBUG_MODE:
                    print(f"✗ Serial write error: {e}")
                return False
        
        return False
    
    def _format_data(self, data_dict: Dict[str, Any]) -> str:
        """
        Format data into Arduino-readable string
        
        Format: <MODE:value,H:value,V:value,D:value,M:value>\n
        """
        mode = data_dict.get('mode', 'NONE')
        horizontal = data_dict.get('horizontal', 'NONE')
        vertical = data_dict.get('vertical', 'NONE')
        distance = data_dict.get('distance', 'NONE')
        match = data_dict.get('match', 'NO')
        
        # Additional position data (optional, for advanced control)
        x_pos = data_dict.get('x', 0)
        y_pos = data_dict.get('y', 0)
        
        # Format as structured string
        data_string = (
            f"<MODE:{mode},"
            f"H:{horizontal},"
            f"V:{vertical},"
            f"D:{distance},"
            f"M:{match},"
            f"X:{x_pos},"
            f"Y:{y_pos}>\n"
        )
        
        return data_string
    
    def send_command(self, command: str) -> bool:
        """
        Send a simple command to Arduino
        
        Args:
            command: Command string (e.g., "STOP", "RESET")
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected or not self.serial_connection:
            if config.DEBUG_MODE:
                print(f"⚠ Cannot send command '{command}': Not connected")
            return False
        
        try:
            message = f"<CMD:{command}>\n"
            self.serial_connection.write(message.encode('utf-8'))
            self.serial_connection.flush()
            
            if config.DEBUG_MODE:
                print(f"→ Command sent: {command}")
            
            return True
        
        except serial.SerialException as e:
            if config.DEBUG_MODE:
                print(f"✗ Command send error: {e}")
            return False
    
    def read_response(self) -> Optional[str]:
        """
        Read response from Arduino
        
        Returns:
            Response string or None
        """
        if not self.is_connected or not self.serial_connection:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                
                if config.DEBUG_MODE:
                    print(f"← Arduino: {response}")
                
                return response
        
        except serial.SerialException as e:
            if config.DEBUG_MODE:
                print(f"✗ Serial read error: {e}")
        
        return None
    
    def close(self):
        """Close serial connection"""
        if self.serial_connection and self.is_connected:
            try:
                self.serial_connection.close()
                self.is_connected = False
                
                if config.DEBUG_MODE:
                    print("✓ Serial connection closed")
            
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"✗ Error closing serial connection: {e}")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to Arduino"""
        if self.is_connected:
            self.close()
        
        if config.DEBUG_MODE:
            print("Attempting to reconnect...")
        
        self._connect()
        return self.is_connected
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baud_rate': self.baud_rate,
            'enabled': config.ENABLE_SERIAL
        }


class DataFormatter:
    """Helper class for formatting different types of tracking data"""
    
    @staticmethod
    def create_tracking_data(mode: str, horizontal: str, vertical: str, 
                           distance: str, match: str, 
                           x: int = 0, y: int = 0) -> Dict[str, Any]:
        """
        Create a properly formatted data dictionary
        
        Args:
            mode: Detection mode (head, eye, mouth, hand, ear)
            horizontal: Horizontal position (LEFT, CENTER, RIGHT)
            vertical: Vertical position (UP, CENTER, DOWN)
            distance: Distance estimate (NEAR, MEDIUM, FAR)
            match: Face match result (YES, NO)
            x: Pixel x-coordinate (optional)
            y: Pixel y-coordinate (optional)
        
        Returns:
            Formatted data dictionary
        """
        return {
            'mode': mode.upper(),
            'horizontal': horizontal.upper(),
            'vertical': vertical.upper(),
            'distance': distance.upper(),
            'match': match.upper(),
            'x': int(x),
            'y': int(y),
            'timestamp': time.time()
        }
    
    @staticmethod
    def create_no_detection_data(mode: str) -> Dict[str, Any]:
        """Create data packet for when no detection occurs"""
        return {
            'mode': mode.upper(),
            'horizontal': 'NONE',
            'vertical': 'NONE',
            'distance': 'NONE',
            'match': 'NO',
            'x': 0,
            'y': 0,
            'timestamp': time.time()
        }