# 🦖 PaleoNet: Dinosaur Species Classifier

![Banner](assets/banner.png)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)](https://www.tensorflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Overview

**PaleoNet** is an advanced dinosaur species classification system built using deep learning. The application can identify 15 different dinosaur species from images with high accuracy, using a convolutional neural network (CNN) based on EfficientNetB0 architecture.

Explore prehistoric creatures through the power of artificial intelligence! 🔍

## ✨ Features

- **Image Classification** - Upload your own dinosaur images for instant species identification
- **Interactive Gallery** - Explore sample images from the test dataset
- **Species Encyclopedia** - Learn fascinating facts about each dinosaur species
- **Model Insights** - Visualize the model architecture and performance metrics
- **Tabbed Navigation** - Switch easily between Home, Upload Image, and Sample Gallery sections

## 🖼️ Application Screenshots

<div align="center">
  <img src="assets/screenshot_main.png" alt="Main Classification Page" width="80%"/>
  <p><i>Main Classification Page: Upload and classify dinosaur images</i></p>
  
  <img src="assets/screenshot_encyclopedia.png" alt="Dinosaur Encyclopedia" width="80%"/>
  <p><i>Dinosaur Encyclopedia: Learn about different dinosaur species</i></p>
  
  <img src="assets/screenshot_model.png" alt="Model Information" width="80%"/>
  <p><i>Model Information: Visualize the model architecture and performance</i></p>
</div>

## 🚀 Quick Start

Install requirements

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run PaleoNet.py
```

Once started, the application will be available at <http://localhost:8501>

## 🦕 Supported Dinosaur Species

PaleoNet can classify the following 15 dinosaur species:

<table>
  <tr>
    <td>🦖 Ankylosaurus</td>
    <td>🦕 Brachiosaurus</td>
    <td>🦎 Compsognathus</td>
  </tr>
  <tr>
    <td>🦖 Corythosaurus</td>
    <td>🦖 Dilophosaurus</td>
    <td>🦎 Dimorphodon</td>
  </tr>
  <tr>
    <td>🦖 Gallimimus</td>
    <td>🦎 Microceratus</td>
    <td>🦖 Pachycephalosaurus</td>
  </tr>
  <tr>
    <td>🦖 Parasaurolophus</td>
    <td>🦖 Spinosaurus</td>
    <td>🦕 Stegosaurus</td>
  </tr>
  <tr>
    <td>🦕 Triceratops</td>
    <td>🦖 Tyrannosaurus Rex</td>
    <td>🦖 Velociraptor</td>
  </tr>
</table>

## 📂 Project Structure

```text
PaleoNet/
├── app/                    # Application code
│   ├── app.py              # Main Streamlit application
│   ├── utils.py            # Utility functions
│   └── pages/              # Additional app pages
│       ├── 01_Model_Info.py
│       └── 02_Dinosaur_Encyclopedia.py
├── assets/                 # Images and static assets
├── data/                   # Dataset directory
│   └── dinosaur_dataset_split/
│       ├── train/          # Training data (70%)
│       ├── val/            # Validation data (15%)
│       └── test/           # Test data (15%)
├── docs/                   # Documentation
├── model/                  # Saved model files
│   ├── dinosaur_classifier_transfer_learning.keras
│   ├── dinosaur_class_mapping.json
│   └── dinosaur_model_performance.json
├── notebooks/              # Jupyter notebooks
│   └── opdracht_CNN_stijnen_simon.ipynb  # Model training notebook
├── requirements.txt        # Project dependencies
├── README.md               # Main documentation
└── LICENSE                 # MIT License
```

## 📊 Model Architecture

PaleoNet uses a transfer learning approach based on EfficientNetB0:

- **Base Model**: EfficientNetB0 pre-trained on ImageNet
- **Feature Extraction**: Global Average Pooling to reduce spatial dimensions
- **Classification Head**:
  - Dense layers (512 & 256 neurons) with ReLU activation
  - Batch Normalization for training stability
  - Dropout (0.4) for regularization
  - Output layer with 15 neurons and softmax activation
- **Training Enhancement**: Data augmentation to improve generalization

The model achieves over 70% accuracy on the test set, with especially strong performance on distinctive species.

## 📝 Documentation

Visit the [docs](docs/) directory for detailed documentation:

- [User Guide](docs/user_guide.md)
- [Model Information](docs/model_info.md)
- [Development Guide](docs/development.md)

## 💻 Streamlit Application

PaleoNet uses [Streamlit](https://streamlit.io/) to create an interactive web application for dinosaur image classification. The application includes:

- **Main Page**: Upload your own images or try sample images for classification
- **Model Info Page**: Visualize the model architecture and performance metrics
- **Dinosaur Encyclopedia**: Learn fascinating facts about each dinosaur species

### Key Application Features

- Real-time classification with confidence scores
- Top-3 prediction display
- Interactive sample gallery with random test images
- Performance metrics visualization
- Detailed dinosaur information cards

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **Dataset**: [Dinosaur Image Dataset from Kaggle](https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species)
- **Technologies**:
  - [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning
  - [EfficientNetB0](https://arxiv.org/abs/1905.11946) architecture by Google
  - [Streamlit](https://streamlit.io/) for the web application interface
  - [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualizations
- **Paleontological Information**: Various academic sources for dinosaur facts and information
- **Development**: Simon Stijnen for the AI Deep Learning course at VIVES University of Applied Sciences (2025)
