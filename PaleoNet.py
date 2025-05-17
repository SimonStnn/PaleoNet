import os
import json
import random
from typing import Any
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications

from utils import download_model_files, download_dataset, display_dino_info

# Set page config
st.set_page_config(
    page_title="PaleoNet - Dinosaur Classifier",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load your model and class mapping
@st.cache_resource
def load_model() -> tuple[tf.keras.Model, dict[str, int], dict[str, Any]]:
    # Define paths
    app_dir = Path(__file__).parent
    model_dir = app_dir / "model"
    data_dir = app_dir / "data"
    model_path = model_dir / "dinosaur_classifier_transfer_learning.keras"
    class_mapping_path = model_dir / "dinosaur_class_mapping.json"
    performance_path = model_dir / "dinosaur_model_performance.json"

    # GitHub release URLs (replace with your actual release URLs)
    github_model_url = (
        "https://github.com/SimonStnn/PaleoNet/releases/download/v0.0.0-pre/model.zip"
    )
    github_dataset_url = (
        "https://github.com/SimonStnn/PaleoNet/releases/download/v0.0.0-pre/dataset.zip"
    )

    # Try to download the model if it doesn't exist
    if not download_model_files(github_model_url, model_dir):
        st.error(
            "Failed to download model files. The app cannot function without these files."
        )
        return None, {}, {}

    # Try to download the dataset if it doesn't exist
    # This is non-blocking - app can still function without the dataset
    if not download_dataset(github_dataset_url, data_dir):
        st.error(
            "Couldn't download the dataset. Some features like the sample gallery may not work correctly."
        )

    try:
        # Load the model
        model = tf.keras.models.load_model(str(model_path))

        # Load the class mapping
        with open(str(class_mapping_path), "r", encoding="utf-8") as f:
            class_mapping: dict[str, int] = json.load(f)

        # Load performance metrics
        with open(str(performance_path), "r", encoding="utf-8") as f:
            performance: dict[str, Any] = json.load(f)

        return model, class_mapping, performance
    except Exception as e:  # type: ignore[broad-exception-caught]
        st.error(f"Error loading model: {e}")
        return None, {}, {}


# Define image size
IMG_SIZE = (296, 296)


# Function to preprocess the image
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    processed_img = applications.efficientnet.preprocess_input(img_batch)
    return processed_img


# Function to make prediction
def predict_species(image_data, model, class_labels):
    processed_img = preprocess_image(image_data)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes = [class_labels[i] for i in top_3_idx]
    top_3_confidences = [predictions[0][i] for i in top_3_idx]

    return predicted_class_idx, confidence, top_3_classes, top_3_confidences


def display_prediction(pred_idx, confidence, top_3_classes, top_3_confidences):
    st.subheader("Prediction")
    st.metric("Predicted species", top_3_classes[0].replace("_", " "))
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # Show top 3 predictions
    st.subheader("Top 3 Predictions:")
    st.dataframe(
        pd.DataFrame(
            {
                "Species": [c.replace("_", " ") for c in top_3_classes],
                "Confidence": [f"{c * 100:.2f}%" for c in top_3_confidences],
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Species": st.column_config.TextColumn(
                "Species", help="Predicted species name"
            ),
            "Confidence": st.column_config.TextColumn(
                "Confidence", help="Confidence level of the prediction"
            ),
        },
    )


# Main app
def main():
    # Sidebar
    st.sidebar.title("PaleoNet 🦖")
    st.sidebar.markdown("A dinosaur species classifier built with TensorFlow")
    st.sidebar.image(
        "https://raw.github.com/SimonStnn/PaleoNet/main/assets/banner.png",
        use_container_width=True,
    )

    # Load model and data
    model, class_mapping, performance = load_model()

    if model is None:
        st.error(
            "Model not loaded. Please check your internet connection or the model files."
        )
        return  # Create class_labels (invert the mapping from class names to indices)
    class_labels = {i: k for i, k in enumerate(class_mapping.keys())}

    # Create tabs for navigation instead of a sidebar selectbox
    # This replaces the previous app_mode selectbox while keeping the multi-page navigation in the sidebar
    home_tab, upload_tab, gallery_tab = st.tabs(
        ["Home", "Upload Image", "Sample Gallery"]
    )

    # Home tab
    with home_tab:
        st.title("PaleoNet: Dinosaur Species Classifier 🦖")

        st.markdown(
            """
        ## Welcome to PaleoNet!
        
        This application uses a Convolutional Neural Network (CNN) to classify 15 different dinosaur species from images.
        
        ### Features:
        
        - Upload your own dinosaur images for classification
        - View sample images from our test set
        - Explore model performance metrics
        
        ### How it Works:
        
        PaleoNet uses transfer learning with EfficientNetB0 as the base model, trained on a dataset of dinosaur images. 
        The model can identify the following species:
        """
        )  # Display available species as a grid
        species_list = list(class_labels.values())
        cols = st.columns(3)
        for i, species in enumerate(species_list):
            cols[i % 3].markdown(f"- {species.replace('_', ' ')}")

        st.markdown(
            """
        ### Get Started:
        
        Use the tabs above to navigate to different sections of the app.
        
        - **Upload Image**: Upload your own dinosaur image for classification
        - **Sample Gallery**: View and classify sample images from our test set
        - Visit the **Model Performance** page in the sidebar pages menu to see how well our model performs
        """
        )

        # Show a random sample image
        st.markdown("### Quick Demo")
        if st.button("Classify a Random Sample"):
            if model is not None:  # Get a random sample from the test set
                test_dir = Path("data") / "dinosaur_dataset_split" / "test"
                # Check if test directory exists
                if not test_dir.exists():
                    st.warning(
                        "Test dataset not available. Please download the dataset from GitHub releases."
                    )
                    return

                species_folders = os.listdir(test_dir)
                random_species = random.choice(species_folders)
                species_folder = test_dir / random_species
                image_files = [
                    f for f in os.listdir(species_folder) if f.endswith(".jpg")
                ]
                random_image = random.choice(image_files)
                image_path = species_folder / random_image

                img = Image.open(image_path)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(
                        img,
                        caption=f"True species: **{random_species.replace('_',' ')}**",
                        use_container_width=True,
                    )

                with col2:
                    with st.spinner("Classifying..."):
                        pred_idx, confidence, top_3_classes, top_3_confidences = (
                            predict_species(img, model, class_labels)
                        )

                    display_prediction(
                        pred_idx, confidence, top_3_classes, top_3_confidences
                    )

                if random_species != top_3_classes[0]:
                    st.error(
                        f"""
                        **Oops!** The model predicted **{top_3_classes[0].replace('_', ' ')}** instead of the true species: **{random_species.replace('_', ' ')}**.
                        """
                    )
                else:
                    st.success(
                        f"""
                        **Success!** The model correctly predicted the species: **{top_3_classes[0].replace('_', ' ')}**.
                        """
                    )

                # Display info about the predicted species
                st.subheader(f"About {top_3_classes[0].replace('_', ' ')}")
                display_dino_info(top_3_classes[0])

    # Upload image tab
    with upload_tab:
        st.title("Upload a Dinosaur Image")

        st.markdown(
            """
        Upload an image of a dinosaur to see what species it might be.
        For best results, use clear images of dinosaur illustrations, models, or reconstructions.
        """
        )

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None and model is not None:
            img = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(img, caption="Uploaded Image", use_container_width=True)

            try:
                with col2:
                    with st.spinner("Classifying..."):
                        pred_idx, confidence, top_3_classes, top_3_confidences = (
                            predict_species(img, model, class_labels)
                        )

                    display_prediction(
                        pred_idx, confidence, top_3_classes, top_3_confidences
                    )

                # Display info about the predicted species
                st.subheader(f"About {top_3_classes[0].replace('_', ' ')}")
                display_dino_info(top_3_classes[0])
            except Exception as e:  # type: ignore[broad-exception-caught]
                st.error(f"Error during prediction: {str(e)[:150]}")

    # Sample gallery tab
    with gallery_tab:
        st.title("Sample Gallery")

        st.markdown(
            """
        Explore sample images from our test dataset and see how the model classifies them.
        """
        )

        if model is not None:
            # Set up paths
            app_dir = Path(__file__).parent
            test_dir = app_dir / "data" / "dinosaur_dataset_split" / "test"

            # Check if test directory exists
            if not test_dir.exists():
                st.warning(
                    "Test dataset not available. Please download the dataset from GitHub releases."
                )
                return

            # Let user select a species
            species_folders = sorted(os.listdir(test_dir))

            selected_species = st.selectbox(
                "Select a dinosaur species:",
                [s.replace("_", " ") for s in species_folders],
            )

            species_folder = test_dir / selected_species.replace(" ", "_")
            image_files = [f for f in os.listdir(species_folder) if f.endswith(".jpg")]

            # Display 3 random images
            if st.button("Show Random Samples"):
                sample_images = random.sample(image_files, min(3, len(image_files)))

                for img_file in sample_images:
                    img_path = species_folder / img_file
                    img = Image.open(img_path)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.image(
                            img, caption=f"Sample: {img_file}", use_container_width=True
                        )

                    with col2:
                        with st.spinner("Classifying..."):
                            pred_idx, confidence, top_3_classes, top_3_confidences = (
                                predict_species(img, model, class_labels)
                            )

                        correct = top_3_classes[0].replace("_", " ") == selected_species

                        st.markdown(f"**True species:** {selected_species}")

                        display_prediction(
                            pred_idx, confidence, top_3_classes, top_3_confidences
                        )

                    if not correct:
                        st.error(
                            f"""
                            **Oops!** The model predicted **{top_3_classes[0].replace('_', ' ')}** instead of the true species: **{selected_species}**.
                            """
                        )
                    else:
                        st.success(
                            f"""
                            **Success!** The model correctly predicted the species: **{top_3_classes[0].replace('_', ' ')}**.
                            """
                        )

                    st.divider()


if __name__ == "__main__":
    main()
