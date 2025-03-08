# Eye Disease Detection Using Self-Supervised Learning & CNNs

## Overview
This project focuses on **early detection of eye diseases** using **Self-Supervised Learning (SSL)** and **Convolutional Neural Networks (CNNs)**. It combines **DINO and SimCLR for feature extraction** with fine-tuned CNN models (**ResNet18, EfficientNet-B0, and DenseNet121**) to improve classification accuracy using **medical imaging datasets**.

## Objectives
- Leverage **Self-Supervised Learning (SSL)** to extract meaningful features from unlabeled eye disease images.
- Train **CNN models** to classify eye diseases using **transfer learning and fine-tuning**.
- Enhance model generalization through **data augmentation and diverse dataset preprocessing**.

## Dataset
- **Source**: [Eye Diseases Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data)
- **Classes**: Normal, Diabetic Retinopathy, Glaucoma, Cataract
- **Size**: 3372 images, categorized into four labeled folders
- **Preprocessing**: Image resizing (256x256), augmentation, and train-validation splitting

## Approach
### 1. **Data Preprocessing & Augmentation**
- Standardized images to **256×256**.
- **Three datasets** created:
  - **Original dataset**
  - **Mildly augmented dataset** (rotation, brightness, shear, horizontal flips)
  - **Heavily augmented dataset** (stronger transformations for robustness)
- Merged datasets for improved generalization.

### 2. **Self-Supervised Learning (SSL)**
- **DINO (Self-Distillation with No Labels)**
  - Trained a **Vision Transformer (ViT)** using **self-supervised learning** to extract features.
  - Fine-tuned a linear classifier for evaluation.
- **SimCLR (Contrastive Learning)**
  - Used a **ResNet-based encoder** and contrastive loss to learn robust embeddings.
  - Created **two augmented views per image** to enforce similarity learning.

### 3. **CNN-Based Classification (Transfer Learning)**
- **Fine-tuned three CNN architectures**:
  - **ResNet18**
  - **EfficientNet-B0**
  - **DenseNet121**
- **Two-stage training approach**:
  1. **Frozen Backbone:** Trained only the final classifier.
  2. **Fine-Tuning:** Unfroze deeper layers for better feature extraction.
- Achieved **highest accuracy (89.55%) with DenseNet121**.

## Results
| Model          | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|------------|--------|----------|
| ResNet18       | 88.95%   | 88.81%     | 88.87% | 88.80%   |
| EfficientNet-B0 | 81.50%  | 81.20%     | 81.40% | 81.17%   |
| DenseNet121    | **89.55%**  | **89.62%** | **89.41%** | **89.48%** |

## Technologies Used
- **Deep Learning & Machine Learning**: PyTorch, TensorFlow, torchvision, scikit-learn
- **Data Processing & Visualization**: NumPy, Pandas, Matplotlib
- **Model Training & Optimization**: Adam optimizer, learning rate scheduling
- **Deployment & Experimentation**: Google Colab, Kaggle

## Installation & Usage
### **1. Clone the Repository**
```bash
 git clone https://github.com/yourusername/Eye-Disease-Detection.git
 cd Eye-Disease-Detection
```

### **2. Install Dependencies**
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### **3. Run Training**
```bash
python train.py  # Replace with actual training script
```

### **4. Evaluate Models**
```bash
python evaluate.py  # Run evaluation script
```

## Future Improvements
- Implement **additional self-supervised learning methods**.
- Optimize hyperparameters and model architectures.
- Deploy as a **real-time eye disease detection API**.

## Contributors
- **Arman Feili**
- **Andrea Melissa Almeida Ortega**
- **Milad Torabi**

## License
This project is licensed under the **MIT License**.

