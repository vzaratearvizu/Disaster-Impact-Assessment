# For file paths and checking if they exist
import os
# For reading the labels files that are in JSON format
import json
# PyTorch main library
import torch
# PyTorch's base class for loading data
from torch.utils.data import Dataset
# For opening and manipulating images
from PIL import Image
# For array operations
import numpy as np
# For preprocessing images before training
from torchvision import transforms

class DisasterDataset(Dataset):
    """
    PyTorch Dataset for loading before and after disaster image pairs
    Performs scene-level damage classification based on worst damage present in the image
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: File path to data directory
            transform: Optional preprocessing step
                       If None, images are returned as raw PIL images
                       If provided, applies the transforms to every image loaded

        """
        # Saves the file path
        self.data_dir = data_dir
        # Creates the full path to the images folder
        self.images_dir = os.path.join(data_dir, 'images')
        # Creates the full path to the labels folder
        self.labels_dir = os.path.join(data_dir, 'labels')
        # Saves the tranforms that were made for __getitem__ to load
        self.transform = transform
        
        # Gets all image pairs
        self.image_pairs = self._get_image_pairs()
        
        # Damage level mapping
        self.damage_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3,
            'un-classified': 0  # Treat un-classified as no-damage
        }
        
    def _get_image_pairs(self):
        """Private method to find all before and after image pairs in the dataset.
        Returns a list of base names (ex: 'hurricane-florence_00000059')"""
        # Stores the image pairs found
        pairs = []
        
        # Get all pre-disaster images
        pre_images = [filename for filename in os.listdir(self.images_dir) 
                     if filename.endswith('_pre_disaster.png')]
        
        # For each pre-disaster image, ccheck if matching post-disaster image exists
        for pre_img in pre_images:
            base_name = pre_img.replace('_pre_disaster.png', '')
            post_img = base_name + '_post_disaster.png'
            
            # Check if post image exists
            if os.path.exists(os.path.join(self.images_dir, post_img)):
                pairs.append(base_name)
        
        # Returns list of all valid image pairs
        return pairs
    
    def _load_damage_label(self, base_name):
        """
        Load damage label from JSON file.
        Returns the overall damage level for the image.
        """
        # Combines labels directory path with filename to post disaster (this is where damage info is)
        label_path = os.path.join(self.labels_dir, 
                                 base_name + '_post_disaster.json')
        
        # Default to no-damage if no label
        if not os.path.exists(label_path):
            return 0
                
        # Reads JSON file and converts to a Python dictionary
        with open(label_path, 'r') as file:
            data = json.load(file)
        
        # Counts how many buildings have each damage level from 
        # 0 (no-damage), 1 (minor), 2 (major), to 3 (destroyed)
        damage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # First checks if structure is found to prevent crashes
        if 'features' in data and 'xy' in data['features']:
            buildings = data['features']['xy']
            
            for building in buildings:
                # First checks if structure is found to prevent crashes
                if 'properties' in building and 'subtype' in building['properties']:
                    # Extracts damage label in text
                    subtype = building['properties']['subtype']
                    # Extracts damage label in integer 
                    # returns 0 if no subtype found
                    damage_level = self.damage_map.get(subtype, 0)
                    # Increases the damage count for that damage level
                    damage_counts[damage_level] += 1
        
        # Determine overall damage level for this image (use the worst damage present)
        if damage_counts[3] > 0:
            return 3  # destroyed
        elif damage_counts[2] > 0:
            return 2  # major-damage
        elif damage_counts[1] > 0:
            return 1  # minor-damage
        else:
            return 0  # no-damage
    
    def __len__(self):
        """Return the number of image pairs"""
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        """
        Get one sample from the dataset by index.
        Returns: (pre_image, post_image, label)
        """
        # Accesses the image_pairs base_name list at the indexed position
        base_name = self.image_pairs[idx]
        
        # Builds the image paths
        pre_path = os.path.join(self.images_dir, base_name + '_pre_disaster.png')
        post_path = os.path.join(self.images_dir, base_name + '_post_disaster.png')
        
        # Loads the image
        pre_img = Image.open(pre_path).convert('RGB')
        post_img = Image.open(post_path).convert('RGB')
        
        # Apply transforms if provided
        # Transforms: resize, convert to tensor, normalize, augment
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
        
        # Load damage label as integer
        label = self._load_damage_label(base_name)
        
        # Returned tuple of (before_disaster, after_disaster, damage_level)
        return pre_img, post_img, label


def get_transforms(image_size=224):
    """
    Get image transformations for training and validation.
    """
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        # Resize for faster training. Large images are slow and memory intensive
        transforms.Resize((image_size, image_size)),
        # Randomly flips left-right to teach model that building can be left or right of an image
        transforms.RandomHorizontalFlip(),  # Flip left-right randomly
        # Randomly flips up-down to teach model that building can be up or down of an image
        transforms.RandomVerticalFlip(),    # Flip up-down randomly
        # Randomly rotates the image to teach model that buildings can be angled differently
        # depending on the angle the satellite took the photo from
        transforms.RandomRotation(10),
        # Randomly adjusts color because satellites take photos at different times of day under
        # different weather conditions
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary colors
        # Converts PIL image to PyTorch Tensor
        transforms.ToTensor(),
        # Normalizes using mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        # Same resize as training
        transforms.Resize((image_size, image_size)),
        # Converts PIL image to PyTorch Tensor
        transforms.ToTensor(),
        # Same normalization as training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tuple containing transformed training and validation images
    return train_transform, val_transform

if __name__ == "__main__":
    print("Testing DisasterDataset...")
    
    # Get only the training transform, ignore validation
    train_transform, _ = get_transforms()

    dataset = DisasterDataset('data/train', transform=train_transform)
    
    print(f"Dataset has been loaded: {len(dataset)} image pairs found")
    
    # Test loading one sample
    pre_img, post_img, label = dataset[0]
    
    print(f"  Sample loaded:")
    print(f"  Pre-image shape: {pre_img.shape}")
    print(f"  Post-image shape: {post_img.shape}")
    print(f"  Label: {label}")
    
    # Count damage distribution
    damage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(len(dataset)):
        # Ignore images, only need the label for counting
        _, _, label = dataset[i]
        damage_counts[label] += 1
    
    print("\n Damage distribution:")
    labels = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for level, count in damage_counts.items():
        percent = (count / len(dataset)) * 100
        print(f"  {labels[level]}: {count} ({percent:.1f}%)")