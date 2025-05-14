import os
import json
import random
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from typing import Any
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications

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
    try:
        model = tf.keras.models.load_model(
            "model/dinosaur_classifier_transfer_learning.keras"
        )

        # Load the class mapping
        with open("model/dinosaur_class_mapping.json", "r", encoding="utf-8") as f:
            class_mapping: dict[str, int] = json.load(f)

        # Load performance metrics
        with open("model/dinosaur_model_performance.json", "r", encoding="utf-8") as f:
            performance: dict[str, Any] = json.load(f)

        return model, class_mapping, performance
    except Exception as e:  # type: ignore
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
    st.table(
        pd.DataFrame(
            {
                "Species": [c.replace("_", " ") for c in top_3_classes],
                "Confidence": [f"{c * 100:.2f}%" for c in top_3_confidences],
            }
        )
    )


# Display dinosaur info
def display_dino_info(species):
    dino_info = {
        "Ankylosaurus": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "8 meters",
            "weight": "4-8 tons",
            "description": "Ankylosaurus was a heavily armored dinosaur with a club-like tail that it could swing as a weapon.",
        },
        "Brachiosaurus": {
            "period": "Late Jurassic",
            "diet": "Herbivore",
            "length": "25 meters",
            "weight": "30-50 tons",
            "description": "Brachiosaurus was a massive sauropod with a long neck that allowed it to feed on tall trees.",
        },
        "Compsognathus": {
            "period": "Late Jurassic",
            "diet": "Carnivore",
            "length": "1 meter",
            "weight": "3 kilograms",
            "description": "Compsognathus was one of the smallest dinosaurs, about the size of a chicken.",
        },
        "Corythosaurus": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "9 meters",
            "weight": "3-4 tons",
            "description": "Corythosaurus had a distinctive hollow crest on its head that may have been used for vocalizations.",
        },
        "Dilophosaurus": {
            "period": "Early Jurassic",
            "diet": "Carnivore",
            "length": "7 meters",
            "weight": "400 kilograms",
            "description": "Dilophosaurus had two crests on its head and, contrary to 'Jurassic Park', did not spit venom.",
        },
        "Dimorphodon": {
            "period": "Early Jurassic",
            "diet": "Carnivore (Fish)",
            "length": "1 meter wingspan",
            "weight": "1-2 kilograms",
            "description": "Dimorphodon was a flying reptile with a large head and two types of teeth.",
        },
        "Gallimimus": {
            "period": "Late Cretaceous",
            "diet": "Omnivore",
            "length": "6 meters",
            "weight": "400 kilograms",
            "description": "Gallimimus was an ostrich-like dinosaur and one of the fastest dinosaurs.",
        },
        "Microceratus": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "0.6 meters",
            "weight": "3 kilograms",
            "description": "Microceratus was a small ceratopsian with a tiny frill and no horns.",
        },
        "Pachycephalosaurus": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "4.5 meters",
            "weight": "450 kilograms",
            "description": "Pachycephalosaurus had a thick, domed skull that it may have used for head-butting.",
        },
        "Parasaurolophus": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "10 meters",
            "weight": "2.5 tons",
            "description": "Parasaurolophus had a long, hollow crest on its head that may have been used for vocalizations.",
        },
        "Spinosaurus": {
            "period": "Mid Cretaceous",
            "diet": "Carnivore (Fish)",
            "length": "15-18 meters",
            "weight": "7-20 tons",
            "description": "Spinosaurus had a sail-like structure on its back and was likely semi-aquatic.",
        },
        "Stegosaurus": {
            "period": "Late Jurassic",
            "diet": "Herbivore",
            "length": "9 meters",
            "weight": "5-7 tons",
            "description": "Stegosaurus had large plates along its back and spikes on its tail called thagomizer.",
        },
        "Triceratops": {
            "period": "Late Cretaceous",
            "diet": "Herbivore",
            "length": "9 meters",
            "weight": "5-9 tons",
            "description": "Triceratops had three horns on its face and a large frill that protected its neck.",
        },
        "Tyrannosaurus_Rex": {
            "period": "Late Cretaceous",
            "diet": "Carnivore",
            "length": "12 meters",
            "weight": "8-14 tons",
            "description": "Tyrannosaurus Rex was one of the largest land carnivores with powerful jaws and tiny arms.",
        },
        "Velociraptor": {
            "period": "Late Cretaceous",
            "diet": "Carnivore",
            "length": "2 meters",
            "weight": "15-20 kilograms",
            "description": "Velociraptor was much smaller than shown in 'Jurassic Park' and had feathers.",
        },
    }

    species_name = species.replace("_", " ")
    if species in dino_info:
        info = dino_info[species]

        st.subheader(f"About {species_name}")

        st.table(pd.DataFrame.from_dict(info, orient="index"))


# Main app
def main():
    # Sidebar
    st.sidebar.title("PaleoNet 🦖")
    st.sidebar.markdown("A dinosaur species classifier built with TensorFlow")
    st.sidebar.image(
        "https://raw.github.com/SimonStnn/PaleoNet/main/assets/banner.png",
        use_container_width=True,
    )

    app_mode = st.sidebar.selectbox(
        "Choose Mode", ["Home", "Upload Image", "Sample Gallery"]
    )  # Load model and data
    model, class_mapping, performance = load_model()

    # Create class_labels (invert the mapping from class names to indices)
    class_labels = {i: k for i, k in enumerate(class_mapping.keys())}

    # Home page
    if app_mode == "Home":
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
        
        Use the sidebar to navigate to different sections of the app.
        
        - **Upload Image**: Upload your own dinosaur image for classification
        - **Sample Gallery**: View and classify sample images from our test set
        - Visit the **Model Performance** page in the sidebar pages menu to see how well our model performs
        """
        )

        # Show a random sample image
        st.markdown("### Quick Demo")
        if st.button("Classify a Random Sample"):
            if model is not None:
                # Get a random sample from the test set
                test_dir = os.path.join("data", "dinosaur_dataset_split", "test")
                species_folders = os.listdir(test_dir)
                random_species = random.choice(species_folders)
                species_folder = os.path.join(test_dir, random_species)
                image_files = [
                    f for f in os.listdir(species_folder) if f.endswith(".jpg")
                ]
                random_image = random.choice(image_files)
                image_path = os.path.join(species_folder, random_image)

                img = Image.open(image_path)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(
                        img,
                        caption=f"True species: **{random_species}**",
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
                display_dino_info(top_3_classes[0])

    # Upload image page
    elif app_mode == "Upload Image":
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
                display_dino_info(top_3_classes[0])
            except Exception as e:  # type: ignore[broad-exception-caught]
                st.error(f"Error during prediction: {str(e)[:150]}")

    # Sample gallery page
    elif app_mode == "Sample Gallery":
        st.title("Sample Gallery")

        st.markdown(
            """
        Explore sample images from our test dataset and see how the model classifies them.
        """
        )

        if model is not None:
            # Let user select a species
            test_dir = os.path.join("data", "dinosaur_dataset_split", "test")
            species_folders = sorted(os.listdir(test_dir))

            selected_species = st.selectbox(
                "Select a dinosaur species:",
                [s.replace("_", " ") for s in species_folders],
            )

            species_folder = os.path.join(test_dir, selected_species.replace(" ", "_"))
            image_files = [f for f in os.listdir(species_folder) if f.endswith(".jpg")]

            # Display 3 random images
            if st.button("Show Random Samples"):
                sample_images = random.sample(image_files, min(3, len(image_files)))

                for img_file in sample_images:
                    img_path = os.path.join(species_folder, img_file)
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
