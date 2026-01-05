# For file paths and checking if they exist
import os
# For finding duration of training
import time
# PyTorch main library
import torch
# PyTorch neural network module
import torch.nn as nn
# Algorithms that update model weights for adaptive learning rates
import torch.optim as optim
# Manages parallel loading, shuffling, and data batches
from torch.utils.data import DataLoader, random_split
# Progress bar with time
from tqdm import tqdm
# Plotting library
import matplotlib.pyplot as plt
# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import DisasterDataset, get_transforms
from model import get_model

start_time = time.time()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    model: neural network (DamageClassifier)
    dataloader: provides batches of data, handles shuffling
    criterion: loss function
    optimizer: updates model weights
    device: run on cpu or gpu
    """
    model.train()
    # Accumulates loss from batches and averaged at end
    running_loss = 0.0
    # How many predictions were correct
    correct = 0
    # Total predications made
    total = 0
    # Loops through all batches
    for pre_img, post_img, labels in tqdm(dataloader, desc="Training"):
        # Move data to device (CPU or GPU)
        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        labels = labels.to(device)
        
        # Zero gradient to prevent accumulation from previous batch
        # How much to change each weight, calculated during backpropagation
        optimizer.zero_grad()
        
        # Forward pass
        # Pass image through model, extract features, returns predictions (logits)
        outputs = model(pre_img, post_img)
        # Lower = better with 0 = perfect predicition
        # Compares predicitons to true labels, returning error score
        loss = criterion(outputs, labels)
        
        # Backward pass (back propagation)
        # Calculates gradients for all weights and goes backward to figure out how
        # to adjust weights to reduce loss
        loss.backward()
        # Takes gradient calculation from .backward() and updates weights to improve model
        optimizer.step()
        
        # Statistics
        # Converts PyTorch tensor to Python number
        # Averages over all batches later to give average loss for the epoch
        running_loss += loss.item()
        # predicted is class/damage level prediction
        # Dim = 1 to compare classes. Dim = 0 would compare batches
        # Score value not needed, only which class prediction one
        _, predicted = torch.max(outputs.data, 1)
        # Counts the total number of images processed in each batch in the epoch
        total += labels.size(0)
        # Counts how many predictions were correct
        correct += (predicted == labels).sum().item()
    
    # Average loss across all batches
    epoch_loss = running_loss / len(dataloader)
    # Percentage of correct predictions
    epoch_acc = 100 * correct / total
    
    # Tuple of average loss in epoch and percent correct
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    # Turns off dropout, freezes batch normalizations, and
    # predictions are deterministic
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # Saves all predictions
    all_preds = []
    # Saves all true labels
    all_labels = []
    
    # Disables gradient tracking
    with torch.no_grad():
        for pre_img, post_img, labels in tqdm(dataloader, desc="Validating"):
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            labels = labels.to(device)
            
            outputs = model(pre_img, post_img)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Saves predictions for validation
            # Change to numpy as confusion matrix expects numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # Tuple with loss, accuracy, predictions, and labels
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history.png'):
    """
    Plot training and validation metrics
    train_losses: list of training losses per epoch
    train_accs: list of training accuracies
    val_losses: list of validation losses
    val_accs: list of valiation accuracies
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    labels: True image labels
    preds: Model predictions
    """
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Damage', 'Minor', 'Major', 'Destroyed'],
                yticklabels=['No Damage', 'Minor', 'Major', 'Destroyed'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def train_model(
    data_dir='data/train',
    num_epochs=10,
    batch_size=16,
    learning_rate=0.001,
    val_split=0.2,
    save_dir='models'
):
    """
    Main training function
    
    Args:
        data_dir: Path to training data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_split: Fraction of data for validation
        model_version: 'v1' or 'v2'
        save_dir: Directory to save models
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    # train_transform is with augmentation, val_transform has no augmentation
    train_transform, val_transform = get_transforms(image_size=224)
    
    # Load full dataset
    full_dataset = DisasterDataset(data_dir, transform=train_transform)
    
    # Split into train and validation
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset to use validation transforms
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    
    # Create model
    print("\nCreating model...")
    model = get_model(num_classes=4, pretrained=True)
    model = model.to(device)
    
    # Loss function with class weights (handle imbalance)
    # Adjust these based on the class distribution
    class_weights = torch.tensor([1.0, 6.8, 3.1, 1.4]).to(device)
    # Measures how wrong predictions are
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Adam = Adaptive Movement Estimation
    # Adjusts leaarning rate per parameter
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
    
    # Training history
    train_losses = [] # Loss per epoch
    train_accs = [] # Accuracy per epoch
    val_losses = [] 
    val_accs = []
    
    # Track best model in case it gets worse in later epochs
    best_val_acc = 0.0
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n")
        
        # Train
        # Loops through all batches, prediction is made, loss is calculated,
        # backpropagation, weights are updated, returns average loss and accuracy
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        # Loops through all batches, model is set to eval mode, prediction with no
        # gradient tracking, loss is calculated, stats are tracked, prediction is saved
        # for confusion matrix, returns loss, accuracy, prredictions, and labels
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        # Reduceed learning rate by half if loss has not improved in recent epochs
        scheduler.step(val_loss)
        
        # Save history to plot training curve
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth')) # .pth is a PyTorch file
            print(f"Saved new best model (Val Acc: {val_acc:.2f}%)")
    
    print("Training complete!")
    print(f"Duration: {int((time.time() - start_time)//60)} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
   # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs,
                         save_path=os.path.join(save_dir, 'training_history.png'))
    
    # Load best model before final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Final evaluation on validation set
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    # Precision shows how accurate it was at making predicitons
    # Recall shows how many of a category was identified
    # f1-score is the harmonic mean of precision and recall, the overall performance
    # Support shows how many examples of a class are in a validation set
    class_names = ['No Damage', 'Minor', 'Major', 'Destroyed']
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(val_labels, val_preds,
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Save classification report to text file
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(val_labels, val_preds, target_names=class_names))
    print(f"Classification report saved to {report_path}")
    
    # Trained model that can be saved for later use or used for making predictions
    return model

# Line 241 class weights
if __name__ == "__main__":
    # Train the model
    model = train_model(
        data_dir='data/train', # Where data is located
        num_epochs=14, # How many times to run through dataset
        batch_size=16, # Amount of images per batch
        learning_rate=0.001, # Step size for weight updates
        val_split=0.2, # 20% for validation
        save_dir='models' # Where to save results
    )