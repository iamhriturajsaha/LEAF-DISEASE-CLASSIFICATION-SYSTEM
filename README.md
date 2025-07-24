# 🌿Leaf Disease Classification System

A deep learning-based system for automated identification and classification of plant leaf diseases using Convolutional Neural Networks (CNN). This project helps agricultural professionals, researchers and plant enthusiasts quickly and accurately diagnose plant diseases from leaf images using the PlantVillage dataset.

![Demo](https://github.com/shukur-alom/leaf-diseases-detect/blob/main/Media/website.gif)

## 🌱 Features

- **Multi-class Classification** - Identifies multiple types of leaf diseases from the PlantVillage dataset
- **Custom CNN Architecture** - Built with a robust convolutional neural network with batch normalization and dropout layers
- **Data Augmentation** - Enhanced training with rotation, shifting, shearing and zooming techniques
- **Web Interface** - User-friendly Streamlit application for easy image upload and prediction
- **Real-time Prediction** - Fast inference for immediate disease diagnosis
- **Agricultural Focus** - Designed specifically for agricultural and horticultural applications

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamhriturajsaha/LEAF-DISEASE-CLASSIFICATION-SYSTEM.git
   cd LEAF-DISEASE-CLASSIFICATION-SYSTEM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 📋 Requirements

The project dependencies are listed in `requirements.txt`. Key libraries include -
- TensorFlow/Keras for deep learning model development
- OpenCV for image processing and manipulation
- NumPy for numerical computations
- Scikit-learn for data preprocessing and model evaluation
- Matplotlib for visualization and plotting
- Streamlit for web interface development
- Pillow for additional image handling

## 🏗️ Model Architecture

- **Custom CNN Architecture** - Multi-layer convolutional neural network built from scratch
- **Input Processing** - 256x256 RGB images with normalization
- **Network Layers** -
  - Multiple Conv2D layers with ReLU activation
  - Batch normalization for training stability
  - MaxPooling for feature reduction
  - Dropout layers for regularization prevention
  - Dense layers for final classification
- **Training Dataset** - PlantVillage dataset with organized plant disease categories
- **Data Augmentation** - Rotation, shifting, shearing, zooming and horizontal flipping
- **Optimization** - Adam optimizer with learning rate decay
- **Training Parameters** - 25 epochs, batch size of 32, initial learning rate of 0.001

## 📊 Dataset Information

The model is trained on the **PlantVillage dataset**, which contains -
- High-quality images of plant leaves with various diseases
- Organized folder structure by plant type and disease category
- Images resized to 256x256 pixels for consistent processing
- Limited to 200 images per disease category for balanced training
- Support for JPG image formats

## 🔧 Usage

### Web Interface
1. Launch the Streamlit application
2. Upload an image of a diseased leaf
3. View the classification results with confidence scores
4. Get recommendations based on the detected disease

## 🎯 Performance

- **Training Configuration** - 25 epochs with early stopping capabilities
- **Data Split** - 80% training, 20% testing
- **Image Processing** - 256x256 pixel resolution with normalization
- **Model Size** - Compact CNN architecture suitable for deployment
- **Accuracy** - Evaluated on test set with comprehensive metrics

## 🔮 Future Enhancements

- [ ] Mobile application development
- [ ] Real-time camera integration
- [ ] Treatment recommendations
- [ ] Multi-language support
- [ ] Batch processing capabilities
- [ ] API endpoint development
