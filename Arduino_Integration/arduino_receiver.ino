/*
 * Arduino Vision Tracking Receiver
 * Receives data from Python vision system and controls motors/actuators
 * 
 * Data Format: <MODE:value,H:value,V:value,D:value,M:value,X:value,Y:value>
 * Example: <MODE:HEAD,H:LEFT,V:CENTER,D:NEAR,M:YES,X:320,Y:240>
 */

// Pin definitions (adjust based on your hardware)
#define MOTOR_H_LEFT_PIN 3    // Horizontal motor - left direction
#define MOTOR_H_RIGHT_PIN 5   // Horizontal motor - right direction
#define MOTOR_V_UP_PIN 6      // Vertical motor - up direction
#define MOTOR_V_DOWN_PIN 9    // Vertical motor - down direction
#define LED_MATCH_PIN 13      // LED for face match indicator
#define BUZZER_PIN 11         // Buzzer for alerts

// Motor speed constants
#define MOTOR_SPEED_FAST 200
#define MOTOR_SPEED_MEDIUM 150
#define MOTOR_SPEED_SLOW 100
#define MOTOR_STOP 0

// Data structure for tracking information
struct TrackingData {
  String mode;
  String horizontal;
  String vertical;
  String distance;
  String match;
  int x;
  int y;
  bool valid;
};

TrackingData currentData;
String inputBuffer = "";
unsigned long lastDataTime = 0;
const unsigned long DATA_TIMEOUT = 2000; // 2 seconds timeout

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize motor pins
  pinMode(MOTOR_H_LEFT_PIN, OUTPUT);
  pinMode(MOTOR_H_RIGHT_PIN, OUTPUT);
  pinMode(MOTOR_V_UP_PIN, OUTPUT);
  pinMode(MOTOR_V_DOWN_PIN, OUTPUT);
  
  // Initialize indicator pins
  pinMode(LED_MATCH_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Stop all motors
  stopAllMotors();
  
  // Startup indication
  digitalWrite(LED_MATCH_PIN, HIGH);
  delay(500);
  digitalWrite(LED_MATCH_PIN, LOW);
  
  Serial.println("Arduino Vision Tracker Ready");
  Serial.println("Waiting for data...");
}

void loop() {
  // Read incoming serial data
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    
    if (inChar == '<') {
      // Start of new message
      inputBuffer = "";
    }
    else if (inChar == '>') {
      // End of message - parse it
      parseData(inputBuffer);
      lastDataTime = millis();
      inputBuffer = "";
    }
    else {
      // Add to buffer
      inputBuffer += inChar;
    }
  }
  
  // Check for data timeout
  if (millis() - lastDataTime > DATA_TIMEOUT && currentData.valid) {
    // No data received - stop tracking
    stopAllMotors();
    currentData.valid = false;
    Serial.println("WARNING: Data timeout - stopping motors");
  }
  
  // Execute tracking commands if data is valid
  if (currentData.valid) {
    executeTracking();
  }
  
  // Small delay for stability
  delay(20);
}

void parseData(String data) {
  // Expected format: MODE:value,H:value,V:value,D:value,M:value,X:value,Y:value
  
  currentData.valid = false;
  
  // Split by comma
  int modeIdx = data.indexOf("MODE:");
  int hIdx = data.indexOf(",H:");
  int vIdx = data.indexOf(",V:");
  int dIdx = data.indexOf(",D:");
  int mIdx = data.indexOf(",M:");
  int xIdx = data.indexOf(",X:");
  int yIdx = data.indexOf(",Y:");
  
  if (modeIdx == -1 || hIdx == -1 || vIdx == -1 || dIdx == -1 || mIdx == -1) {
    Serial.println("ERROR: Invalid data format");
    return;
  }
  
  // Extract values
  currentData.mode = data.substring(modeIdx + 5, hIdx);
  currentData.horizontal = data.substring(hIdx + 3, vIdx);
  currentData.vertical = data.substring(vIdx + 3, dIdx);
  currentData.distance = data.substring(dIdx + 3, mIdx);
  
  if (xIdx != -1) {
    currentData.match = data.substring(mIdx + 3, xIdx);
    currentData.x = data.substring(xIdx + 3, yIdx).toInt();
    currentData.y = data.substring(yIdx + 3).toInt();
  } else {
    currentData.match = data.substring(mIdx + 3);
    currentData.x = 0;
    currentData.y = 0;
  }
  
  currentData.valid = true;
  
  // Update match indicator
  if (currentData.match == "YES") {
    digitalWrite(LED_MATCH_PIN, HIGH);
  } else {
    digitalWrite(LED_MATCH_PIN, LOW);
  }
  
  // Debug output
  Serial.print("Parsed - H:");
  Serial.print(currentData.horizontal);
  Serial.print(" V:");
  Serial.print(currentData.vertical);
  Serial.print(" D:");
  Serial.print(currentData.distance);
  Serial.print(" M:");
  Serial.println(currentData.match);
}

void executeTracking() {
  // Stop all motors first
  stopAllMotors();
  
  // Check if detection is valid
  if (currentData.horizontal == "NONE") {
    return; // No target detected
  }
  
  // Calculate motor speed based on distance
  int motorSpeed = MOTOR_SPEED_MEDIUM;
  if (currentData.distance == "NEAR") {
    motorSpeed = MOTOR_SPEED_SLOW;
  } else if (currentData.distance == "FAR") {
    motorSpeed = MOTOR_SPEED_FAST;
  }
  
  // Horizontal tracking
  if (currentData.horizontal == "LEFT") {
    analogWrite(MOTOR_H_LEFT_PIN, motorSpeed);
    analogWrite(MOTOR_H_RIGHT_PIN, 0);
  }
  else if (currentData.horizontal == "RIGHT") {
    analogWrite(MOTOR_H_LEFT_PIN, 0);
    analogWrite(MOTOR_H_RIGHT_PIN, motorSpeed);
  }
  else if (currentData.horizontal == "CENTER") {
    analogWrite(MOTOR_H_LEFT_PIN, 0);
    analogWrite(MOTOR_H_RIGHT_PIN, 0);
  }
  
  // Vertical tracking
  if (currentData.vertical == "UP") {
    analogWrite(MOTOR_V_UP_PIN, motorSpeed);
    analogWrite(MOTOR_V_DOWN_PIN, 0);
  }
  else if (currentData.vertical == "DOWN") {
    analogWrite(MOTOR_V_UP_PIN, 0);
    analogWrite(MOTOR_V_DOWN_PIN, motorSpeed);
  }
  else if (currentData.vertical == "CENTER") {
    analogWrite(MOTOR_V_UP_PIN, 0);
    analogWrite(MOTOR_V_DOWN_PIN, 0);
  }
  
  // Alert on face match
  if (currentData.match == "YES") {
    tone(BUZZER_PIN, 1000, 100); // Short beep
  }
}

void stopAllMotors() {
  analogWrite(MOTOR_H_LEFT_PIN, 0);
  analogWrite(MOTOR_H_RIGHT_PIN, 0);
  analogWrite(MOTOR_V_UP_PIN, 0);
  analogWrite(MOTOR_V_DOWN_PIN, 0);
}

// Emergency stop function (can be called from serial command)
void emergencyStop() {
  stopAllMotors();
  digitalWrite(LED_MATCH_PIN, LOW);
  noTone(BUZZER_PIN);
  Serial.println("EMERGENCY STOP ACTIVATED");
}
