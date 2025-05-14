import os
import json
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(page_title="PaleoNet - Model Info", page_icon="🦖", layout="wide")

# Get the absolute path to the root directory
app_dir = Path(__file__).parent
root_dir = app_dir.parent
model_dir = root_dir / "model"

st.title("Model Information")

st.markdown(
    """
## Model Architecture

The dinosaur classification model uses transfer learning with EfficientNetB0 as the base model. This approach leverages a pre-trained model that has already learned general image features from a large dataset (ImageNet) and fine-tunes it for our specific task of dinosaur classification.

### Architecture Details:

1. **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
   - Efficient architecture designed to maximize accuracy for a given parameter budget
   - Good balance between model size and performance

2. **Feature Extraction**:
   - Global Average Pooling to reduce spatial dimensions
   
3. **Classification Head**:
   - 512-neuron Dense layer with ReLU activation
   - Batch Normalization for training stability
   - Dropout (0.4) for regularization
   - 256-neuron Dense layer with ReLU activation 
   - Batch Normalization
   - Dropout (0.4)
   - Output layer with 15 neurons (one per species) and softmax activation
"""
)


# Create a simplified diagram of the model architecture
def create_model_diagram():
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove axes
    ax.axis("off")

    # Define the positions and sizes
    blocks = [
        {"name": "Input\n(296x296x3)", "x": 0.1, "width": 0.15, "color": "lightblue"},
        {
            "name": "EfficientNetB0\n(Base Model)",
            "x": 0.3,
            "width": 0.25,
            "color": "lightgreen",
        },
        {
            "name": "Global Average\nPooling",
            "x": 0.6,
            "width": 0.15,
            "color": "lightsalmon",
        },
        {
            "name": "Dense (512)\nBatchNorm\nDropout",
            "x": 0.8,
            "width": 0.15,
            "color": "lightpink",
        },
        {
            "name": "Dense (256)\nBatchNorm\nDropout",
            "x": 1.0,
            "width": 0.15,
            "color": "lightpink",
        },
        {"name": "Output\n(15 classes)", "x": 1.2, "width": 0.15, "color": "khaki"},
    ]

    # Draw the blocks
    for block in blocks:
        ax.add_patch(
            plt.Rectangle(
                (block["x"], 0.4),
                block["width"],
                0.2,
                facecolor=block["color"],
                edgecolor="black",
                alpha=0.7,
            )
        )
        ax.text(
            block["x"] + block["width"] / 2,
            0.5,
            block["name"],
            ha="center",
            va="center",
            fontsize=10,
            rotation=90,
        )

    # Draw the arrows connecting the blocks
    for i in range(len(blocks) - 1):
        ax.arrow(
            blocks[i]["x"] + blocks[i]["width"],
            0.5,
            blocks[i + 1]["x"] - (blocks[i]["x"] + blocks[i]["width"]),
            0,
            head_width=0.02,
            head_length=0.01,
            fc="black",
            ec="black",
        )

    plt.title("Model Architecture", fontsize=14)

    return fig


st.pyplot(create_model_diagram())

st.markdown(
    """
## Training Process

The model was trained in two phases:

1. **Initial Training (Feature Extraction)**:
   - Base model (EfficientNetB0) weights were frozen
   - Only the top classification layers were trained
   - Learning rate: 1e-3
   - This allowed the model to learn dinosaur-specific features while keeping the pre-trained weights

2. **Fine-tuning**:
   - Last 10% of the base model layers were unfrozen
   - The entire model was trained with a lower learning rate (1e-5)
   - Class weights were applied to handle imbalance in the dataset
   - This allowed the model to adapt the pre-trained features to our specific domain

### Training Techniques:

- **Data Augmentation**: Applied to increase the effective size of the training set
  - Rotation, width/height shifts, zooming, horizontal flips
  - Brightness variation and shearing

- **Early Stopping**: Prevented overfitting by monitoring validation accuracy

- **Learning Rate Reduction**: Automatically reduced when validation loss plateaued

- **Class Weights**: Applied to handle imbalanced classes in the dataset
"""
)


def plot_training_history():
    # Load training history from csv file
    chart_data = pd.read_csv(
        os.path.join(model_dir, "training_history.csv"),
        names=[
            "Epoch",
            "Training Accuracy",
            "Validation Accuracy",
            "Training Loss",
            "Validation Loss",
        ],
        header=0,  # Use the first row as header
    )
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(
            chart_data.iloc[:, 1:3],  # Show columns 1 and 2
            use_container_width=True,
        )

    with col2:
        st.line_chart(
            chart_data.iloc[:, 3:5],  # Show columns 3 and 4
            use_container_width=True,
        )


st.subheader("Training History")
plot_training_history()

st.markdown(
    """
## Dataset Information

The model was trained on the [Dinosaur Image Dataset](https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species) from Kaggle, which includes images of 15 dinosaur species.

### Dataset Preparation:
- Split into train (70%), validation (15%), and test (15%) sets
- Images resized to 296x296 pixels
- Data augmentation applied to the training set to increase diversity

### Species Distribution:
The dataset contains varying numbers of images per species, which created a class imbalance addressed during training with class weights.
"""
)


# Create a bar chart of the class distribution using actual data
def plot_class_distribution():
    try:
        # Load performance data using the absolute path
        performance_path = os.path.join(model_dir, "dinosaur_model_performance.json")
        with open(performance_path, "r", encoding="utf-8") as f:
            performance = json.load(f)

        # Extract species names and counts
        species = list(performance["classes"].keys())
        counts = list(performance["classes"].values())

    except Exception as e:  # type: ignore[broad-exception-caught]
        st.error(f"Error loading class distribution: {str(e)}")
        # Fallback to default values if loading fails
        species = []
        counts = []

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create the bar chart
    bars = ax.bar(species, counts)

    # Customize the plot
    ax.set_title("Number of Images per Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Images")
    ax.set_xticks(range(len(species)))  # Set the tick positions
    ax.set_xticklabels(species, rotation=45, ha="right")  # Set the tick labels

    # Add the counts on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{height}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    return fig


st.pyplot(plot_class_distribution())

st.markdown(
    """
## Model Evaluation

The model was evaluated on a held-out test set to assess its performance on unseen data.

The key metrics include:
- **Accuracy**: How often the model predicts the correct dinosaur species
- **Precision**: The ability of the model to correctly identify only relevant instances
- **Recall**: The ability of the model to find all relevant instances
- **F1-Score**: The harmonic mean of precision and recall
"""
)

# Load performance metrics if available
try:
    # Load performance data using the absolute path
    performance_path = os.path.join(model_dir, "dinosaur_model_performance.json")
    with open(performance_path, "r") as f:
        performance = json.load(f)

    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{performance['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{performance['precision']*100:.2f}%")
    col3.metric("Recall", f"{performance['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{performance['f1_score']*100:.2f}%")
except FileNotFoundError:
    st.error("Performance metrics file not found.")

st.markdown(
    """
### Observations:

- The model performs best on species with distinctive features like Ankylosaurus (armor plates) and Tyrannosaurus Rex
- More challenging classifications include similar-looking species like some of the bipedal carnivores
- Image quality and viewpoint significantly impact classification accuracy

### Future Improvements:

1. **Data Augmentation**: Add more diverse augmentation techniques
2. **Architecture**: Experiment with different base models and hyperparameters
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Dataset Expansion**: Add more images, especially for underrepresented species
"""
)

st.markdown(
    """
---
### About the Developer

This dinosaur classification model and application were developed by Simon Stijnen for the AI Deep Learning course at VIVES University of Applied Sciences (May 2025).
"""
)
