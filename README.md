# Face Emotion Detection with InceptionV3

## Project Overview

This capstone project implements **Face Emotion Detection** systems using Transfer Learning with the **InceptionV3** convolutional neural network architecture. The project contains multiple implementations exploring different approaches to classify facial emotions into 7 categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

The project utilizes pre-trained InceptionV3 models from ImageNet weights and fine-tunes them for emotion classification tasks on the FER (Facial Expression Recognition) dataset.

---

## Project Structure

```
Capstone/
├── InceptionNet_v3_Face_Emotion_Detection.ipynb   # Main emotion detection model
├── P. inception v3.ipynb                          # Alternative CNN implementation
├── dataset/                                        # Dataset directory
└── README.md                                       # This file
```

---

## Notebooks Description

### 1. **InceptionNet_v3_Face_Emotion_Detection.ipynb**
Main implementation of the emotion detection system using transfer learning.

**Key Features:**
- **Dataset**: Downloads FER (Facial Expression Recognition) dataset from Kaggle
- **Pre-trained Model**: InceptionV3 with ImageNet weights
- **Input Shape**: 160x160x3 RGB images
- **Architecture**: 
  - InceptionV3 base model (frozen layers)
  - Flatten layer
  - Dense layer (1024 units, ReLU activation)
  - Dropout (0.2)
  - Output layer (7 units, sigmoid activation for multi-class emotion classification)
- **Training**: 50 epochs with RMSprop optimizer
- **Loss Function**: Binary crossentropy
- **Metric**: Accuracy
- **Data Augmentation**: ImageDataGenerator with rescaling (1/255)

**Main Capabilities:**
- Trains on train/test split of emotion dataset
- Generates training/validation accuracy and loss plots
- Performs real-time emotion detection from photo inputs
- Includes face detection using Haar Cascade classifier
- Saves trained model as `emotion_model.h5`

**Usage Flow:**
1. Downloads and extracts dataset from Kaggle
2. Loads and preprocesses images
3. Trains InceptionV3 model on emotion classification
4. Visualizes training metrics
5. Tests model on captured/uploaded photos
6. Provides emotion probability distribution

---

### 2. **P. inception v3.ipynb**
Alternative implementation with custom CNN architecture and enhanced metrics tracking.

**Key Features:**
- **Data Source**: Google Drive integration for larger datasets
- **Custom CNN Model**: 
  - Conv2D layer (8 filters, 3x3 kernel)
  - MaxPooling2D
  - BatchNormalization
  - Dropout (0.2)
  - Conv2D layer (32 filters, 3x3 kernel)
  - MaxPooling2D
  - Flatten
  - Dense layer (512 units, ReLU)
  - Output layer (1 unit, sigmoid)
- **Input Shape**: 180x180x3 RGB images
- **Training Parameters**:
  - Batch size: 32
  - Epochs: 50
  - Optimizer: Adam (learning rate: 0.001)
  - Validation split: 30%

**Advanced Metrics:**
- Accuracy
- AUC (ROC curve)
- Precision
- Recall
- True Positives/Negatives
- False Positives/Negatives

**Data Processing:**
- Data augmentation (random flip, rotation, zoom)
- Normalization (rescaling to 0-1 range)
- Image dataset loading with TensorFlow
- Train-validation split

---

## Technologies & Libraries

### Core Deep Learning
- **TensorFlow/Keras**: Model building and training
- **Pre-trained Models**: InceptionV3

### Data Processing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **OpenCV (cv2)**: Face detection and image processing
- **PIL**: Image handling

### Visualization
- **Matplotlib**: Training metrics and results visualization
- **Seaborn**: Enhanced statistical visualizations

### Machine Learning Utilities
- **scikit-learn**: Confusion matrix and classification reports
- **Keras ImageDataGenerator**: Data augmentation

### Environment
- **Google Colab**: Cloud-based GPU support
- **Kaggle API**: Dataset downloading

---

## Dataset

**Source**: Emotion Detection FER Dataset (Kaggle)

**Structure**:
```
dataset/
├── train/           # Training images (7 emotion categories)
└── test/            # Test/validation images
```

**Classes** (7 emotions):
1. Angry
2. Disgusted
3. Fearful
4. Happy
5. Neutral
6. Sad
7. Surprised

**Image Specifications**:
- Format: RGB (3 channels)
- Resolution: 160x160 or 180x180 pixels
- Total training images: ~28,709
- Total test images: ~7,178

---

## Model Architecture Comparison

### InceptionNet v3 (Transfer Learning)
| Component | Details |
|-----------|---------|
| Base Model | InceptionV3 (pretrained on ImageNet) |
| Frozen Layers | Yes |
| Input Size | 160×160×3 |
| Trainable Layers | Custom top layers only |
| Parameters | ~23.8M (most pre-trained) |
| Training Time | Faster (pre-trained features) |

### Custom CNN
| Component | Details |
|-----------|---------|
| Architecture | Sequential CNN |
| Trainable | All layers |
| Input Size | 180×180×3 |
| Convolutional Filters | 8, 32 |
| Total Parameters | ~100K+ |
| Training Time | Moderate |

---

## Results & Outputs

### Training Metrics
Both models generate:
- **Accuracy Plots**: Training vs Validation accuracy over epochs
- **Loss Plots**: Training vs Validation loss over epochs
- **Combined Plots**: All metrics on single visualization

### Model Outputs
- **emotion_model.h5**: Trained emotion detection model
- **Emotion Predictions**: Probability distribution across 7 emotion classes
- **Bar Chart Visualization**: Emotion probabilities with labels

### Real-time Detection
- Face detection using Haar Cascade classifier
- Emotion prediction from detected faces
- Visual results display

---

## Requirements

### Python Environment
```
Python 3.7+
TensorFlow >= 2.0
Keras
OpenCV (cv2)
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
```

### Hardware
- **GPU**: Recommended (CUDA-enabled GPU for faster training)
- **RAM**: Minimum 8GB
- **Storage**: ~2GB for dataset + models

### Platforms
- **Google Colab**: Primary development environment
- **Local Machine**: Python 3.7+ with GPU support
- **Jupyter Notebook**: Compatible environment

---

## Installation & Setup

### For Google Colab:
```python
# Install Kaggle
!pip install kaggle

# Upload kaggle.json credentials
from google.colab import files
files.upload()

# Setup Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d ananthu017/emotion-detection-fer
```

### For Local Environment:
```bash
# Create virtual environment
python -m venv emotion_detection_env
source emotion_detection_env/bin/activate  # On Windows: emotion_detection_env\Scripts\activate

# Install dependencies
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn

# Download dataset manually or use Kaggle API
kaggle datasets download -d ananthu017/emotion-detection-fer
unzip emotion-detection-fer.zip
```

---

## Usage Guide

### Training the Model

**Option 1: Transfer Learning (InceptionV3)**
```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model

# Load pre-trained InceptionV3
base_model = InceptionV3(input_shape=(160, 160, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(7, activation='sigmoid')(x)

model = Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

**Option 2: Custom CNN**
```python
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Conv2D(8, (3, 3), padding='same', activation='relu', 
                  input_shape=(180, 180, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Making Predictions

```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load trained model
emotion_model = load_model('emotion_model.h5')

# Prepare image
img = image.load_img('photo.jpg', color_mode="rgb", target_size=(160, 160))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

# Predict emotions
predictions = emotion_model.predict(x)

# Map to emotion labels
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
for emotion, confidence in zip(emotions, predictions[0]):
    print(f"{emotion}: {confidence:.2%}")
```

---

## Performance Metrics

### Expected Results
- **Training Accuracy**: 80-90%
- **Validation Accuracy**: 75-85%
- **Test Accuracy**: 70-80%

*Note: Actual results depend on dataset size, training duration, and hyperparameter tuning*

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Kaggle dataset download fails | Verify kaggle.json is in ~/.kaggle/ with correct permissions |
| Face detection doesn't work | Ensure haarcascade_frontalface_alt.xml is available in working directory |
| Out of memory errors | Reduce batch size or image dimensions |
| Poor accuracy | Increase epochs, add data augmentation, fine-tune learning rate |
| Slow training | Use GPU (Colab GPU runtime), reduce model complexity |

---

## Future Enhancements

- [ ] Real-time emotion detection from webcam feed
- [ ] Ensemble models for improved accuracy
- [ ] Attention mechanisms for face region focus
- [ ] Multi-face detection and emotion recognition
- [ ] REST API deployment
- [ ] Mobile application integration
- [ ] Emotion trends analysis over time
- [ ] Integration with facial recognition systems

---

## References

- **InceptionV3 Paper**: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- **Transfer Learning**: [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **FER Dataset**: [Facial Expression Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
- **OpenCV Documentation**: [OpenCV Face Detection](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)

---

## Author & Project Info

**Project**: Face Emotion Detection Capstone  
**Date**: 2026  
**Environment**: Google Colab / Jupyter Notebook  
**Dataset**: FER (Facial Expression Recognition)  

---

## License

This project is educational and provided as-is for capstone/research purposes.

---

## Contact & Support

For issues or questions about the project, refer to:
- Kaggle FER Dataset discussion
- TensorFlow/Keras documentation
- Google Colab support

---

**Last Updated**: April 2026
