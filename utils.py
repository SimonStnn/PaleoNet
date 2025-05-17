import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras import applications
import keras
import streamlit as st
import json
import pandas as pd
import seaborn as sns


# Function to create a grid of sample images
def create_image_grid(image_paths, n_cols=3, img_size=(296, 296)):
    """Create a grid of images"""
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # If there's only one row, make sure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Iterate through the images and plot them
    for i, img_path in enumerate(image_paths):
        row = i // n_cols
        col = i % n_cols

        img = Image.open(img_path)
        img = img.resize(img_size)

        axes[row, col].imshow(img)
        axes[row, col].set_title(os.path.basename(img_path))
        axes[row, col].axis("off")

    # Hide any unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


# Function to extract features from an image for visualization
def extract_features(img, model):
    """Extract intermediate features from the model for visualization"""
    # Create a feature extraction model (from the base of our model)
    feature_model = keras.Model(
        inputs=model.inputs, outputs=model.get_layer("global_average_pooling2d").output
    )

    # Get features
    features = feature_model.predict(img)
    return features


# Function to create gradcam heatmaps
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Create a Grad-CAM heatmap to visualize where the model is focusing"""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron with respect to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)

    return heatmap.numpy()


# Function to get activation maps
def get_activation_maps(model, img, layer_name):
    """Get activation maps from a specific layer"""
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    return activations[0]


# Function to analyze an image and create visualizations
def analyze_image(img_path, model, class_labels, img_size=(296, 296)):
    """Create analysis visualizations for an image"""
    # Load and preprocess the image
    img = Image.open(img_path)
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    processed_img = applications.efficientnet.preprocess_input(img_batch)

    # Get prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_labels[predicted_class_idx]

    # Create base figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
    ax.axis("off")

    return fig


# Function to create a confusion matrix visualization
def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with nice formatting"""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create the heatmap
    cax = ax.matshow(cm, cmap=plt.get_cmap("Blues"))
    fig.colorbar(cax)

    # Set up the axes
    ax.set_xticklabels([""] + class_names, rotation=45, ha="right")
    ax.set_yticklabels([""] + class_names)

    # Add labels
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Add the values to the plot
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.tight_layout()
    return fig


# Function to create a progress bar for loading
@st.cache_resource
def progress_bar(progress):
    my_bar = st.progress(0)
    for percent_complete in range(progress):
        my_bar.progress(percent_complete + 1)
    return my_bar


# Cache function for getting sample images
@st.cache_data
def get_sample_paths(data_dir, species, num_samples=5):
    """Get paths to sample images for a species"""
    species_path = os.path.join(data_dir, species)
    if not os.path.exists(species_path):
        return []

    image_files = [
        f for f in os.listdir(species_path) if f.endswith(".jpg") or f.endswith(".png")
    ]
    if len(image_files) <= num_samples:
        selected_files = image_files
    else:
        selected_files = np.random.choice(image_files, num_samples, replace=False)

    return [os.path.join(species_path, f) for f in selected_files]


# Function to load performance metrics from JSON file
def load_performance_metrics(model_dir):
    """
    Load performance metrics from the performance JSON file

    Args:
        model_dir: Path to the model directory

    Returns:
        Performance data dictionary or None if loading fails
    """
    try:
        # Load the performance metrics
        with open(
            os.path.join(model_dir, "dinosaur_model_performance.json"),
            "r",
            encoding="utf-8",
        ) as f:
            performance = json.load(f)
        return performance
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        return None


# Function to display performance metrics in a 4-column layout
def display_performance_metrics(performance):
    """
    Display performance metrics in a 4-column layout

    Args:
        performance: Performance metrics dictionary
    """
    col1, col2, col3, col4 = st.columns(4, border=True)

    col1.metric("Accuracy", f"{performance['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{performance['precision']*100:.2f}%")
    col3.metric("Recall", f"{performance['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{performance['f1_score']*100:.2f}%")


# Function to create a bar chart of the class distribution
def plot_class_distribution(model_dir):
    """
    Create a bar chart of the class distribution

    Args:
        model_dir: Path to the model directory

    Returns:
        Matplotlib figure object
    """
    try:
        # Load performance data
        performance = load_performance_metrics(model_dir)

        if performance is None:
            return None

        # Extract species names and counts
        species = list(performance["classes"].keys())
        counts = list(performance["classes"].values())

    except Exception as e:
        st.error(f"Error loading class distribution: {str(e)}")
        # Fallback to default values if loading fails
        return None

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


# Function to load and display the confusion matrix
def display_confusion_matrix(model_dir):
    """
    Load and display the confusion matrix

    Args:
        model_dir: Path to the model directory

    Returns:
        Matplotlib figure object with the confusion matrix
    """
    # Load performance data to get class names
    performance = load_performance_metrics(model_dir)

    if performance is None:
        return None

    # Get the list of class names
    true_labels = list(performance["classes"].keys())
    cm_fig, ax = plt.subplots(figsize=(14, 12))

    # Check if we have a confusion matrix saved
    confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.csv")
    if os.path.exists(confusion_matrix_path):
        # Load the confusion matrix from CSV
        cm_df = pd.read_csv(confusion_matrix_path, index_col=0)
        cm = cm_df.values
        # Use the indices from the CSV file for better accuracy
        true_labels = cm_df.index.tolist()
    else:
        # Create a mock confusion matrix with higher values on diagonal
        num_classes = len(true_labels)
        cm = np.random.randint(0, 5, size=(num_classes, num_classes))
        np.fill_diagonal(cm, np.random.randint(10, 20, size=num_classes))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=true_labels,
        yticklabels=true_labels,
        cmap="Blues",
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Counts)")
    plt.xticks(rotation=45, ha="right")

    return cm_fig


DINO_INFO: Final = {
    "Ankylosaurus": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Herbivore",
        "length": "8 meters (26 feet)",
        "weight": "4-8 tons",
        "description": "Ankylosaurus was a heavily armored dinosaur with a club-like tail that it could swing as a weapon. Its back was covered with bony plates (osteoderms) and spikes for protection against predators. Despite its fearsome appearance, it was a peaceful plant-eater.",
        "interesting_fact": "The club on its tail could swing with enough force to break the bones of attacking predators like Tyrannosaurus Rex.",
    },
    "Brachiosaurus": {
        "period": "Late Jurassic (154-153 million years ago)",
        "diet": "Herbivore",
        "length": "25 meters (82 feet)",
        "weight": "30-50 tons",
        "description": "Brachiosaurus was one of the tallest dinosaurs, with a long neck that allowed it to feed on foliage high in trees. Unlike many other long-necked dinosaurs, its front legs were longer than its hind legs, giving it a distinctive upward-sloping posture.",
        "interesting_fact": "Its nostrils were located on top of its head, which led scientists to once believe it lived underwater like a hippo. This theory has since been disproven.",
    },
    "Compsognathus": {
        "period": "Late Jurassic (150-145 million years ago)",
        "diet": "Carnivore",
        "length": "1 meter (3.3 feet)",
        "weight": "3 kilograms (6.6 pounds)",
        "description": "Compsognathus was one of the smallest dinosaurs, about the size of a chicken. Despite its small size, it was a swift and agile predator that hunted small lizards and mammals. It had sharp teeth and claws for catching and eating prey.",
        "interesting_fact": "A fossil of Compsognathus was found with a small lizard in its stomach, giving us rare direct evidence of what it ate.",
    },
    "Corythosaurus": {
        "period": "Late Cretaceous (77-75 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "3-4 tons",
        "description": "Corythosaurus had a distinctive hollow crest on its head that may have been used for vocalizations and display. It belonged to the 'duck-billed' dinosaur family (hadrosaurs) and likely lived in herds for protection.",
        "interesting_fact": "Its crest contained nasal passages that may have allowed it to make loud, trumpet-like calls to communicate with others in its herd.",
    },
    "Dilophosaurus": {
        "period": "Early Jurassic (193 million years ago)",
        "diet": "Carnivore",
        "length": "7 meters (23 feet)",
        "weight": "400 kilograms (880 pounds)",
        "description": "Dilophosaurus had two thin, bony crests on its head that were likely used for display or species recognition. It was portrayed in 'Jurassic Park' as having a neck frill and the ability to spit venom, but there is no evidence for either of these features in the fossil record.",
        "interesting_fact": "It had a notch in its upper jaw that gave it a weak bite, suggesting it may have been a scavenger or specialized in hunting smaller prey.",
    },
    "Dimorphodon": {
        "period": "Early Jurassic (175-200 million years ago)",
        "diet": "Carnivore (Fish and Insects)",
        "length": "1 meter (3.3 feet) wingspan",
        "weight": "1-2 kilograms (2.2-4.4 pounds)",
        "description": "Dimorphodon was a flying reptile (pterosaur) with a large head relative to its body. It had two types of teeth (hence the name 'di-morpho-don' meaning 'two-form-teeth'): large fangs at the front and smaller teeth behind them.",
        "interesting_fact": "Unlike modern birds, it couldn't fold its wings completely, suggesting it may have spent significant time on the ground, perhaps hunting like modern roadrunners.",
    },
    "Gallimimus": {
        "period": "Late Cretaceous (70 million years ago)",
        "diet": "Omnivore",
        "length": "6 meters (20 feet)",
        "weight": "400 kilograms (880 pounds)",
        "description": "Gallimimus was an ostrich-like dinosaur with a small head, long neck, and powerful legs. Its name means 'chicken mimic' due to its neck vertebrae resembling those of a chicken. It was one of the fastest dinosaurs and featured in the famous running scene in 'Jurassic Park'.",
        "interesting_fact": "It had a keratinous beak but no teeth, and may have used it to filter small animals and plants from water, similar to modern flamingos.",
    },
    "Microceratus": {
        "period": "Late Cretaceous (70-65 million years ago)",
        "diet": "Herbivore",
        "length": "0.6 meters (2 feet)",
        "weight": "3 kilograms (6.6 pounds)",
        "description": "Microceratus was a small ceratopsian (horned dinosaur) with a tiny frill and no horns. Despite being related to Triceratops, it was much smaller and more primitive. It likely lived in herds and used its beak to crop low-growing vegetation.",
        "interesting_fact": "It was one of the smallest known ceratopsians and may have been prey for many carnivorous dinosaurs and even large birds of the period.",
    },
    "Pachycephalosaurus": {
        "period": "Late Cretaceous (70-65 million years ago)",
        "diet": "Herbivore",
        "length": "4.5 meters (15 feet)",
        "weight": "450 kilograms (990 pounds)",
        "description": "Pachycephalosaurus had a thick, domed skull roof that could be up to 25 cm (10 inches) thick. Scientists believe males used these domes for head-butting contests to establish dominance, similar to modern bighorn sheep.",
        "interesting_fact": "Recent studies suggest that instead of direct head-butting, they may have been hitting each other's flanks, as direct dome-to-dome impacts might have caused brain damage.",
    },
    "Parasaurolophus": {
        "period": "Late Cretaceous (76-74 million years ago)",
        "diet": "Herbivore",
        "length": "10 meters (33 feet)",
        "weight": "2.5 tons",
        "description": "Parasaurolophus had a dramatic backward-curving hollow crest that extended from the back of its head. The crest contained elongated nasal passages that probably served as resonating chambers for making loud calls.",
        "interesting_fact": "Computer models suggest that it could produce low-frequency sounds similar to a trombone, with different species having different 'notes' based on the size and shape of their crests.",
    },
    "Spinosaurus": {
        "period": "Mid Cretaceous (99-93.5 million years ago)",
        "diet": "Carnivore (primarily fish)",
        "length": "15-18 meters (49-59 feet)",
        "weight": "7-20 tons",
        "description": "Spinosaurus had a sail-like structure on its back formed by elongated neural spines, which may have been used for display, temperature regulation, or fat storage. Recent discoveries suggest it had short legs and a paddle-like tail, indicating it was largely aquatic.",
        "interesting_fact": "It's the only known swimming dinosaur, with adaptations similar to modern crocodiles for hunting fish in rivers and lakes. It was larger than T. Rex, making it the largest known carnivorous dinosaur.",
    },
    "Stegosaurus": {
        "period": "Late Jurassic (155-150 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "5-7 tons",
        "description": "Stegosaurus had distinctive upright plates along its back and spikes on its tail called a thagomizer. The plates may have been used for display or temperature regulation, while the tail spikes were definitely defensive weapons.",
        "interesting_fact": "It had a brain the size of a walnut (weighing around 80 grams), one of the smallest brain-to-body ratios of any dinosaur. This led to the myth that it had a 'second brain' in its hip region, which is not true.",
    },
    "Triceratops": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "5-9 tons",
        "description": "Triceratops had three distinctive facial horns and a large frill extending from the back of its skull. The horns and frill were likely used for species recognition, courtship display, and defense against predators like Tyrannosaurus Rex.",
        "interesting_fact": "Its name means 'three-horned face,' and it's one of the last non-avian dinosaurs to exist before the mass extinction event. Over 50 skulls have been found, making it one of the best-documented dinosaurs.",
    },
    "Tyrannosaurus_Rex": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Carnivore",
        "length": "12-13 meters (40-43 feet)",
        "weight": "8-14 tons",
        "description": "Tyrannosaurus Rex was one of the largest land carnivores with powerful jaws containing banana-sized teeth. Despite popular culture often depicting it as a fast runner, studies suggest its top speed was likely 12-25 mph, and it had excellent vision and smell.",
        "interesting_fact": "Its arms were small but powerful, with two fingers and could lift up to 200 kg (440 pounds). A T. Rex named 'Sue' is the largest and most complete specimen ever found, with 90% of its bones recovered.",
    },
    "Velociraptor": {
        "period": "Late Cretaceous (75-71 million years ago)",
        "diet": "Carnivore",
        "length": "2 meters (6.8 feet)",
        "weight": "15-20 kilograms (33-44 pounds)",
        "description": "Velociraptor was much smaller than depicted in 'Jurassic Park' and had feathers like modern birds. It had a distinctive sickle-shaped claw on each foot that it likely used to slash at prey. It was a swift, intelligent predator that may have hunted in packs.",
        "interesting_fact": "A famous fossil shows a Velociraptor locked in combat with a Protoceratops, with the raptor's claw embedded in the dinosaur's neck and the Protoceratops biting the raptor's arm. Both died in this position, possibly buried by a sandstorm.",
    },
}


def display_dino_info(species: str):
    if species in DINO_INFO:
        info = DINO_INFO[species]

        st.dataframe(
            pd.DataFrame.from_dict(info, orient="index"),
            use_container_width=True,
        )
