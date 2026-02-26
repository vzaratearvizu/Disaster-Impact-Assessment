# Disaster Impact Assessment from Satellite Imagery

*Deep learning system for scene-level disaster damage classification using satellite image comparison*

![Training Results](experiments/experiment_23_data/confusion_matrix.png)

## 🎯 Overview

This project uses convolutional neural networks to assess disaster impact severity by comparing satellite imagery taken before and after natural disasters. The system performs **scene-level classification**, identifying areas by damage severity to support disaster response prioritization and resource allocation.

**Achievement:** 67% validation accuracy with 77% F1-score on undamaged area detection and 71% on destroyed zone identification through systematic experimentation.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU (recommended) or CPU
- 50GB disk space for dataset

### Installation
```bash
# Clone repository
git clone https://github.com/vzaratearvizu/disaster-impact-assessment.git
cd disaster-impact-assessment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Register at [xView2 Dataset](https://xview2.org/dataset)
2. Download the training set (~7.8GB compressed)
3. Extract to `data/train/` directory

### Train the Model
```bash
python src/train.py
```

**Training time:** ~28 minutes on GPU, ~40-50 minutes on CPU

**Output:** Trained model saved to `models/best_model.pth` with training curves and confusion matrix

## 📊 Results

### Best Performance

**Model:** ResNet18-based Siamese CNN  
**Validation Accuracy:** 67%  
**Training Time:** ~28 minutes (NVIDIA GPU)

### Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Damage | 0.75 | 0.79 | **0.77** | 249 |
| Minor Damage | 0.00 | 0.00 | 0.00 | 38 |
| Major Damage | 0.46 | 0.37 | 0.41 | 86 |
| Destroyed | 0.65 | 0.79 | **0.71** | 186 |
| **Overall** | - | - | **0.67** | **559** |

### Training Progress

![Training History](experiments/experiment_23_data/training_history.png)

*Loss and accuracy curves showing model convergence over 14 training epochs*

## 🔬 Experimental Process

Conducted **29 systematic experiments** exploring:
- Class weighting strategies (uniform, inverse frequency, domain-priority)
- Model architectures (ResNet18 vs ResNet50)
- Training duration (10-22 epochs)
- Learning rates (0.001-0.002)
- Data augmentation techniques

**Key Findings:**
1. **Uniform class weighting** [1.0, 1.0, 1.0, 1.0] performs optimally, contradicting theoretical inverse frequency weighting
2. **ResNet18 outperforms ResNet50** on this dataset size due to reduced overfitting
3. **Saturation augmentation degrades performance** on satellite imagery
4. **Optimal training duration:** 12-14 epochs before overfitting occurs
5. **Minor damage class unfixable** with only 153 training examples (6.8% of dataset)

📖 See [experiments.md](experiments.md) for complete experimental log with all 29 experiments documented.

## 🛠️ Technical Architecture

### Model Design

**Siamese CNN with Shared Feature Extractor:**
```
Before Image (224×224×3)
    ↓
ResNet18 Feature Extractor (shared weights)
    ↓
Features (512) ──┐
                 ├─→ Concatenate (1024 features)
Features (512) ──┘        ↓
    ↓              Fully Connected Classifier
ResNet18 Feature Extractor     ↓
    ↓              3 layers with dropout
After Image (224×224×3)        ↓
                    Damage Classification (4 classes)
```

**Architecture Components:**
- **Backbone:** Pre-trained ResNet18 (ImageNet weights)
- **Feature Fusion:** Concatenation of before/after features
- **Classifier:** 
  - Layer 1: 1024 → 512 (Dropout 50%)
  - Layer 2: 512 → 256 (Dropout 30%)
  - Layer 3: 256 → 4 (output)

### Optimal Hyperparameters

- **Training epochs:** 14 (best validation at epoch 12)
- **Batch size:** 16
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss function:** CrossEntropyLoss with uniform class weights [1.0, 1.0, 1.0, 1.0]
- **Data augmentation:** Random horizontal/vertical flips, color jitter (brightness±20%, contrast±20%)

## 📁 Project Structure
```
disaster-impact-assessment/
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset for image pairs and labels
│   ├── model.py            # Siamese CNN architecture definition
│   └── train.py            # Training loop, validation, and evaluation
├── experiments/
│   ├── experiment_1_data/  # Baseline experiment results
│   ├── experiment_23_data/ # Best performance experiment
│   └── ...                 # 29 total experiments with visualizations
├── notebooks/
│   ├── explore_data_1.ipynb # Data exploration and analysis
|   └── explore_data_2.ipynb # Data exploration and analysis
├── experiments.md          # Complete log of all 29 experiments
├── README.md
└── requirements.txt
```

## 📚 Dataset

### xBD (xView2) Satellite Imagery Dataset

- **Source:** [xView2 Challenge](https://xview2.org/dataset)
- **Size:** 2,799 satellite image pairs
- **Image Resolution:** 1024×1024 pixels (resized to 224×224 for training)
- **Disasters:** 10 types including hurricanes, earthquakes, floods, wildfires, tsunamis, volcanoes
- **Classes:** No-damage (46.4%), Minor (6.8%), Major (14.8%), Destroyed (32.0%)
- **Split:** 80% training (2,240 images), 20% validation (559 images)
- **License:** CC BY-NC-SA 4.0

### Class Distribution Challenge

The dataset exhibits severe class imbalance, particularly for minor damage (only 6.8% of examples). This proved to be an unfixable limitation with only 153 training examples, even with aggressive class weighting strategies.

## 🎯 Applications

This technology supports:

- **Emergency Response Teams:** Rapid damage assessment for resource allocation
- **Insurance Companies:** Automated claims processing and damage verification
- **Government Agencies:** Recovery planning and infrastructure assessment  
- **NGOs:** Humanitarian aid prioritization in disaster zones
- **Research:** Climate change impact studies and disaster pattern analysis

## ⚙️ How It Works

### Data Pipeline

1. **Image Pair Loading:** Scans directory for matching before/after disaster images
2. **Label Extraction:** Parses JSON files containing building damage annotations
3. **Labeling Strategy:** Uses "worst damage present" approach - if any building in the scene is destroyed, the entire image is labeled as destroyed
4. **Preprocessing:** 
   - Resize from 1024×1024 to 224×224
   - Normalize using ImageNet statistics
   - Apply data augmentation (training only)

### Model Training

1. **Feature Extraction:** Both images processed through shared ResNet18 backbone
2. **Feature Fusion:** Concatenate 512-dimensional features from each image
3. **Classification:** 3-layer fully connected network predicts damage level
4. **Optimization:** Adam optimizer minimizes CrossEntropyLoss
5. **Validation:** Evaluate on unseen images after each epoch
6. **Model Selection:** Save best model based on validation accuracy

### Inference
```python
import torch
from src.model import get_model
from src.dataset import get_transforms

# Load trained model
model = get_model(num_classes=4, pretrained=True)
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process images
_, val_transform = get_transforms()
pre_img = val_transform(before_image)
post_img = val_transform(after_image)

# Predict
with torch.no_grad():
    output = model(pre_img.unsqueeze(0), post_img.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)
    
# prediction: 0=No Damage, 1=Minor, 2=Major, 3=Destroyed
```

## 🔍 Scope and Limitations

### What This System Does

- ✅ Classifies overall damage severity in geographic areas
- ✅ Identifies scenes containing destroyed buildings
- ✅ Provides rapid area-wide damage assessment
- ✅ Supports disaster response prioritization

### What This System Does Not Do

- ❌ Identify specific building locations
- ❌ Provide building-by-building damage reports
- ❌ Generate spatial damage maps with coordinates
- ❌ Count or locate individual damaged structures

**Note:** This implementation performs image-level classification rather than instance segmentation. While the xBD dataset includes building polygon annotations, this project uses them to derive scene-level labels based on worst damage present.

## 🚀 Future Improvements

### Technical Enhancements

- **Instance segmentation** for building-level damage localization using polygon annotations
- **Ensemble methods** (averaging 3-5 models for +2-4% accuracy boost)
- **Advanced architectures** (EfficientNet, Vision Transformers)
- **Focal loss** for better class imbalance handling
- **Attention mechanisms** to highlight damaged regions

### Data & Scale

- **Incorporate tier3 and test sets** (~20,000 additional image pairs)
- **Oversample minority classes** to address severe imbalance
- **Multi-scale detection** processing images at different resolutions
- **Temporal sequences** for damage progression analysis

### Deployment

- **Web application** for emergency responders
- **Real-time processing pipeline** for new satellite imagery
- **GIS integration** for spatial visualization
- **Confidence scores** and uncertainty quantification
- **API endpoint** for automated damage assessment

## 📖 Key Learnings

This project demonstrates several important machine learning concepts:

**Transfer Learning Effectiveness**
- Pre-trained ResNet18 features handle class imbalance naturally
- Uniform weighting competitive with complex frequency-based approaches
- ImageNet knowledge transfers effectively to satellite imagery domain

**Dataset Size Considerations**
- 2,799 images insufficient for ResNet50 (25M parameters) - causes overfitting
- ResNet18 (11M parameters) better suited for this dataset scale
- Minority class (153 examples) fundamentally insufficient regardless of weighting

**Systematic Experimentation**
- Theoretical best practices (inverse frequency weighting) don't always work in practice
- Empirical testing reveals simpler approaches often superior
- Importance of validation strategies and overfitting detection

**Domain-Specific Adaptations**
- Satellite imagery requires different augmentation than natural photos
- Color channels have semantic meaning (green=vegetation, blue=water)
- Saturation augmentation creates unrealistic scenarios and degrades performance

## 📊 Comparison to Research

**xView2 Challenge Winners (2019):**
- 1st Place: 79.3% accuracy
- 2nd Place: 76.8% accuracy
- 3rd Place: 74.5% accuracy

**This Project:** 67% accuracy (solo effort, systematic experimentation)

**Gap Analysis:** 8-12% difference attributable to ensemble methods, advanced architectures, larger computational resources, and extended optimization periods available to research teams.

## 🤝 Contributing

This is a portfolio/educational project, but suggestions and feedback are welcome! Feel free to open issues or reach out with questions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [xView2 Challenge](https://xview2.org/) organizers for creating and releasing the xBD dataset
- PyTorch team for the deep learning framework
- DIUx (Defense Innovation Unit) for supporting disaster response research
- ResNet authors for the architecture and pre-trained weights

## 📧 Contact

**Victor Zarate Arvizu**  
zaratearvizuv@gmail.com  
www.linkedin.com/in/victorzaratearvizu  
https://github.com/vzaratearvizu

Project Link: [https://github.com/vzaratearvizu/disaster-impact-assessment](https://github.com/vzaratearvizu/disaster-impact-assessment)

---

⭐ If you found this project helpful or interesting, please consider giving it a star!