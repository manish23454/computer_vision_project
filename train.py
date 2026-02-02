"""
train.py - Train Gesture Recognition Model

This script handles:
- Loading and preprocessing the gesture dataset
- Training the CNN model
- Validation and accuracy metrics
- Saving the trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import time
from model import GestureCNN, get_pretrained_model


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """
    Train the gesture recognition model
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run on (CPU or GPU)
        num_epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    print(f"\nTraining on device: {device}")
    print("="*60)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ==================== TRAINING PHASE ====================
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        
        # ==================== VALIDATION PHASE ====================
        model.eval()  # Set model to evaluation mode
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():  # No gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Statistics
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate validation accuracy
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        # Time taken for this epoch
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Training Loss: {epoch_loss:.4f}")
        print(f"  Training Accuracy: {train_acc:.2f}%")
        print(f"  Validation Accuracy: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"✓ Best model saved with validation accuracy: {best_val_acc:.2f}%\n")
    
    return model, train_losses, val_accuracies


def main():
    """
    Main training pipeline
    """
    # ==================== CONFIGURATION ====================
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    NUM_CLASSES = 5
    TRAIN_SPLIT = 0.8  # 80% training, 20% validation
    
    # Use pretrained model or custom CNN
    USE_PRETRAINED = False  # Set to True to use ResNet18
    
    # Dataset path
    DATASET_PATH = './dataset'
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please ensure your dataset is organized as:")
        print("  dataset/")
        print("    ├── forward/")
        print("    ├── backward/")
        print("    ├── left/")
        print("    ├── right/")
        print("    └── stop/")
        return
    
    # ==================== DEVICE SETUP ====================
    # Use GPU if available (CUDA or MPS for Mac)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\nUsing device: {device}")
    
    # ==================== DATA PREPROCESSING ====================
    # Define data transformations
    # Training transforms include data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),  # Randomly flip images
        transforms.RandomRotation(10),  # Randomly rotate images
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variations
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ==================== DATASET LOADING ====================
    print("\nLoading dataset...")
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Total images: {len(full_dataset)}")
    
    # Check if we have all 5 gesture classes
    if len(class_names) != NUM_CLASSES:
        print(f"\nWARNING: Expected {NUM_CLASSES} classes but found {len(class_names)}")
        print("Please ensure you have folders for: forward, backward, left, right, stop")
    
    # Print class distribution
    print("\nClass distribution:")
    for idx, class_name in enumerate(class_names):
        count = sum(1 for _, label in full_dataset if label == idx)
        print(f"  {class_name}: {count} images")
    
    # ==================== TRAIN/VAL SPLIT ====================
    # Calculate split sizes
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to train and validation sets
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    
    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    # ==================== MODEL INITIALIZATION ====================
    print("\nInitializing model...")
    
    if USE_PRETRAINED:
        print("Using pretrained ResNet18")
        model = get_pretrained_model(num_classes=NUM_CLASSES, pretrained=True)
    else:
        print("Using custom GestureCNN")
        model = GestureCNN(num_classes=NUM_CLASSES)
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== LOSS AND OPTIMIZER ====================
    # CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer with learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=3)
    
    # ==================== TRAINING ====================
    print("\nStarting training...")
    print("="*60)
    
    trained_model, train_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    # ==================== SAVE FINAL MODEL ====================
    print("\nSaving final model...")
    torch.save(trained_model.state_dict(), 'model.pth')
    print("✓ Final model saved as 'model.pth'")
    
    # Save class names for later use
    import json
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print("✓ Class names saved as 'class_names.json'")
    
    # ==================== TRAINING SUMMARY ====================
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print("="*60)
    
    # Check if model meets accuracy threshold
    if max(val_accuracies) >= 80:
        print("\n✓ Model achieved target accuracy (>80%)!")
    else:
        print(f"\n⚠ Model accuracy ({max(val_accuracies):.2f}%) is below target (80%)")
        print("  Consider:")
        print("  - Training for more epochs")
        print("  - Using a pretrained model (set USE_PRETRAINED=True)")
        print("  - Collecting more training data")
        print("  - Adjusting hyperparameters")


if __name__ == "__main__":
    main()