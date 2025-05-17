from pathlib import Path
import pandas as pd
import streamlit as st

from utils import (
    load_performance_metrics,
    display_performance_metrics,
    display_confusion_matrix,
)

# Set page config
st.set_page_config(
    page_title="PaleoNet - Model Performance", page_icon="🦖", layout="wide"
)

# Get the absolute path to the root directory
app_dir = Path(__file__).parent
root_dir = app_dir.parent
model_dir = root_dir / "model"


st.title("Model Performance")

# Load performance metrics
performance = load_performance_metrics(model_dir)

if performance is not None:
    st.subheader("Overall Metrics")
    display_performance_metrics(performance)

    st.markdown(
        """
        ### About the Model
        
        Our model is built using transfer learning with EfficientNetB0 as the base model, which was pre-trained on ImageNet. 
        We fine-tuned the model on our dinosaur dataset with the following steps:
        
        1. Split the dataset into train (70%), validation (15%), and test (15%) sets
        2. Used data augmentation for the training set
        3. First trained only the top layers while keeping the base model frozen
        4. Then fine-tuned the model by unfreezing the last 10% of the base model layers
        5. Used class weights to handle class imbalance
        
        ### Performance Analysis
        
        The model performs well on most species, achieving an overall accuracy of over 80%. 
        Some species are easier to classify than others due to distinctive features.
        
        Common misclassifications occur between:
        
        - Similar body types (e.g., bipedal carnivores)
        - Species with similar head crests or frills
        - Species commonly depicted in similar environments
        """
    )  # Show sample predictions
    st.subheader("Visualization")

    st.markdown("### Per-Class Image Counts")

    # Convert the class counts to a DataFrame for better display
    class_counts = pd.DataFrame(
        list(performance["classes"].items()), columns=["Species", "Image Count"]
    )

    # Display as a bar chart
    st.bar_chart(class_counts.set_index("Species"), use_container_width=True)

    st.markdown(
        """
    ### Class Balance Analysis
    
    The chart above shows the distribution of test images across different dinosaur species.
    A balanced dataset helps ensure that the model performs well across all classes and doesn't
    develop bias toward over-represented classes.
    
    During training, class weights were applied to compensate for any imbalance in the dataset.
    """
    )

    st.markdown("### Confusion Matrix")
    st.markdown(
        """
    The confusion matrix visualizes the model's performance across different classes.
    Each cell in the matrix represents the number of predictions made for a specific true class (rows)
    and predicted class (columns). The diagonal cells represent correct predictions, while off-diagonal
    cells represent misclassifications.
    A well-performing model will have high values on the diagonal and low values elsewhere.
    The confusion matrix can help identify which classes are often confused with each other.
    """
    )

    with st.spinner("Generating confusion matrix..."):
        # Display the confusion matrix using the utility function
        cm_fig = display_confusion_matrix(model_dir)
        if cm_fig is not None:
            st.pyplot(cm_fig)
        else:
            st.error("Could not generate confusion matrix.")
