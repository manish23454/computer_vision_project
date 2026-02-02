"""
model.py - Gesture Recognition CNN Model

This module defines the Convolutional Neural Network architecture
for classifying hand gestures into 5 categories:
forward, backward, left, right, stop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureCNN(nn.Module):
    """
    Convolutional Neural Network for Gesture Recognition
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPooling
    - 2 Fully connected layers
    - Dropout for regularization
    
    Input: RGB images of size 224x224
    Output: 5 class probabilities
    """
    
    def __init__(self, num_classes=5):
        """
        Initialize the CNN architecture
        
        Args:
            num_classes (int): Number of gesture classes (default: 5)
        """
        super(GestureCNN, self).__init__()
        
        # First convolutional block
        # Input: 3x224x224 -> Output: 32x112x112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        # Input: 32x112x112 -> Output: 64x56x56
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        # Input: 64x56x56 -> Output: 128x28x28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        # Input: 128x28x28 -> Output: 256x14x14
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after conv layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        # Final size: 256 * 14 * 14 = 50176
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def get_pretrained_model(num_classes=5, pretrained=True):
    """
    Alternative: Use a pretrained ResNet18 model for transfer learning
    
    This can provide better accuracy with less training time
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained weights
        
    Returns:
        torch.nn.Module: Modified ResNet18 model
    """
    from torchvision import models
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Replace the final fully connected layer
    # ResNet18's fc layer has 512 input features
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


if __name__ == "__main__":
    """
    Test the model architecture
    """
    # Create a sample input tensor
    sample_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    
    # Test custom CNN
    print("Testing Custom GestureCNN:")
    model = GestureCNN(num_classes=5)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n" + "="*50 + "\n")
    
    # Test pretrained ResNet18
    print("Testing Pretrained ResNet18:")
    resnet_model = get_pretrained_model(num_classes=5, pretrained=False)
    output = resnet_model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in resnet_model.parameters())}")