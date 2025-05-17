import os
import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_performance_metrics,
    display_performance_metrics,
    plot_class_distribution,
)


st.set_page_config(page_title="PaleoNet - Model Info", page_icon="🦖", layout="wide")

# Get the absolute path to the root directory
app_dir = Path(__file__).parent
root_dir = app_dir.parent
model_dir = root_dir / "model"

st.title("Model Information")

st.markdown(
    """
## Dataset Information

The model was trained on the [Dinosaur Image Dataset](https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species) from Kaggle, which includes images of 15 dinosaur species.

### Species Distribution:
The dataset contains varying numbers of images per species, which created a class imbalance addressed during training with class weights.

"""
)


# We now use the shared utility function for plotting class distribution

performance = load_performance_metrics(model_dir)

if performance is not None:
    # Convert the class counts to a DataFrame for better display
    class_counts = pd.DataFrame(
        list(performance["classes"].items()), columns=["Species", "Image Count"]
    )

    # Display as a bar chart
    st.bar_chart(class_counts.set_index("Species"), use_container_width=True)

st.markdown(
    """
### Dataset Preparation:
- Split into train (70%), validation (15%), and test (15%) sets
- Images resized to 296x296 pixels
- Data augmentation applied to the training set to increase diversity
"""
)
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
    ]  # Draw the blocks
    for block in blocks:
        ax.add_patch(
            patches.Rectangle(
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
    st.info("Starts fine-tuning at epoch **10**")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.line_chart(
            chart_data.iloc[:, 1:3],  # Show columns 1 and 2
            x_label="Epoch",
            y_label="Accuracy",
            use_container_width=True,
        )

    with col2:
        st.line_chart(
            chart_data.iloc[:, 3:5],  # Show columns 3 and 4
            x_label="Epoch",
            y_label="Loss",
            use_container_width=True,
        )


st.subheader("Training History")
plot_training_history()

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
performance = load_performance_metrics(model_dir)
if performance is not None:
    # Display metrics using the common function
    display_performance_metrics(performance)
else:
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

This dinosaur classification model and application were developed by **Simon Stijnen** for the AI Deep Learning course at VIVES University of Applied Sciences *(May 2025)*.
"""
)
