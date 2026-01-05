# PyTorch main library
import torch
# PyTorch neural network module
import torch.nn as nn
# PyTorch pre-trained neural networks
import torchvision.models as models

class DamageClassifier(nn.Module):
    """
    Siamese CNN network for building damage assessment.
    Classifies overall scene damage severity by comparing satellite image pairs
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        """
        Args:
            num_classes: Number of damage classes (default 4: no-damage, minor, mahor, destroyed)
            pretrained: Use pretrained weights (default: True)
        """
        # Set up internal PyTorch machinery (GPU support, parameter tracking, save/load)
        super(DamageClassifier, self).__init__()
        
        # Use ResNet18 neural network as the feature extractor
        # This is pre-trained on ImageNet
        # Began to use ResNet50 in experiment 26
        resnet = models.resnet18(pretrained=pretrained)
        
        # Get ResNet18's layers minues the classification to use the features for
        # our own classification
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 outputs 512 features. Used to match amount of features to build classifier
        # ResNet50 outputs 2048 features. Used to match amount of features to build classifier
        num_features = 512
        
        # Classifier head
        # Takes concatenated features from before and after images
        # Sqeuential (output 1 is input 2, output 2 is input 3)
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, 512),  # Combine before + after features of images
            # can learn complex patterns since non-linearity is added. Rectified Linear Unit
            nn.ReLU(),
            # Prevents overfitting with 50% chance  to set it to 0
            # No dropout = memorizing training data, poor generalization for new images
            # Yes dropout = learns robust patterns, better perfromance on new images
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            # Deeper layers have less information, so need less chance of dropout to retain the information
            nn.Dropout(0.3),
            # Final layer to 4 damage classifications. Higher logit = higher confidence in accuracy
            nn.Linear(256, num_classes)
        )
        
    def forward(self, pre_img, post_img):
        """
        Forward pass through the network.
        Defines how data flows through the network and automatically called when model is used.
        Args:
            pre_img: Before disaster image tensor [batch_size, 3, 224, 224]
            post_img: After disaster image tensor [batch_size, 3, 224, 224]
            
        Returns:
            predictions: Damage class predictions [batch_size, num_classes]
        """
        # Extract features from pre-disaster image
        pre_features = self.feature_extractor(pre_img)
        # Classifier expects 2D input. Reshape since it can't handle 4D input.
        pre_features = pre_features.view(pre_features.size(0), -1)  # Flatten
        
        # Extract features from post-disaster image
        post_features = self.feature_extractor(post_img)
        post_features = post_features.view(post_features.size(0), -1)  # Flatten
        
        # Concatenate features from both images
        # dim 0 is batch dimension. dim 1 is feature dimension.
        combined_features = torch.cat([pre_features, post_features], dim=1)
        
        # Classify damage level based on found features
        output = self.classifier(combined_features)
        
        # Return damage prediction scores (logits) for each image in batch
        # Shape: [batch_size, num_classes]
        # Each row has 4 scores: [no-damage, minor, major, destroyed]
        # These are raw scores, not probabilities yet
        # During training: used with CrossEntropyLoss (converts to probabilities internally)
        # During inference: use torch.max() to get predicted class
        return output



def get_model(num_classes=4, pretrained=True):
    """
    Makes the damage classifier model.

    Args:
        num_classes: Number of damage classes
        pretrained: Use pretrained weights
    
    Returns:
        model: DamageClassifier
    """
    return DamageClassifier(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test the model
    print("Testing DamageClassifier...")
    
    # Create model
    model = get_model('v1')
    
    # Create dummy input for testing (batch_size=2, channels=3, height=224, width=224)
    pre_img = torch.randn(2, 3, 224, 224)
    post_img = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(pre_img, post_img)
    
    print(f"  Model created successfully")
    print(f"  Input shape: {pre_img.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Only counts parameters that will be updated during training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")