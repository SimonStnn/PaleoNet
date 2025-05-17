# PaleoNet User Guide

This guide will help you get started with using the PaleoNet dinosaur classification application.

## Table of Contents

- [PaleoNet User Guide](#paleonet-user-guide)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
    - [Windows Users](#windows-users)
    - [Manual Startup](#manual-startup)
  - [Using the Application](#using-the-application)
    - [Navigation](#navigation)
    - [Home Tab](#home-tab)
    - [Upload Image Tab](#upload-image-tab)
    - [Sample Gallery Tab](#sample-gallery-tab)
    - [Model Performance](#model-performance)
    - [Model Info](#model-info)
    - [Dinosaur Encyclopedia](#dinosaur-encyclopedia)
  - [Tips for Best Results](#tips-for-best-results)
  - [Troubleshooting](#troubleshooting)

## Installation

Before you can use PaleoNet, you need to install the required dependencies:

1. Ensure you have Python 3.8 or higher installed on your system.

2. Clone the repository or download the project files.

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Windows Users

For Windows users, you can start PaleoNet using PowerShell:

1. Navigate to the project directory
2. Open PowerShell and run:

   ```powershell
   streamlit run PaleoNet.py
   ```

3. The application will start and open in your default web browser

### Manual Startup

If you prefer to start the application manually, you can use the following command:

```bash
streamlit run PaleoNet.py
```

This will launch the Streamlit server and open the application in your default web browser at <http://localhost:8501>.

## Using the Application

### Navigation

The application uses a tabbed interface for the main sections:

- **Home**: Overview of the application
- **Upload Image**: Upload your own images for classification
- **Sample Gallery**: Explore sample images from the test dataset

Additional pages are accessible from the sidebar pages menu.

### Home Tab

The home tab provides an overview of the PaleoNet application, including:

- A brief description of the project and features
- A list of supported dinosaur species
- A quick demo that lets you classify a random sample image

### Upload Image Tab

The "Upload Image" tab allows you to:

1. Upload your own dinosaur image (.jpg, .jpeg, or .png format)
2. View the classification results, including:
   - The predicted species
   - Confidence level
   - Top 3 predictions
   - Information about the predicted dinosaur species

Tips for uploading images:

- Use clear, well-lit images
- Images should show the dinosaur clearly against the background
- Both illustrations and photographs work well

### Sample Gallery Tab

The "Sample Gallery" tab lets you:

1. Select a dinosaur species from the dropdown menu
2. View random sample images from that species
3. See how the model classifies these samples
4. Compare true labels with predictions

This is useful for understanding the model's strengths and weaknesses.

### Model Performance

The "Model Performance" page (accessible from the sidebar) provides:

- Overall accuracy, precision, recall, and F1 score metrics
- Explanation of the model architecture
- Visual representation of the training process
- Insights into where the model performs well and where it struggles

### Model Info

The "Model Info" page (accessible from the sidebar) offers technical details about:

- The model architecture (EfficientNetB0 with custom layers)
- Training process and techniques used
- Dataset preparation and splitting
- Class distribution across the dataset

### Dinosaur Encyclopedia

The "Dinosaur Encyclopedia" page (accessible from the sidebar) is an educational resource that provides:

- Detailed information about each dinosaur species
- Time period when they lived
- Physical characteristics and behavior
- Interesting facts
- Related species and evolutionary relationships

## Tips for Best Results

For the best classification results:

1. Use high-quality images showing the full dinosaur
2. Illustrations and clear renders often work better than photographs
3. Images should have good lighting and minimal background clutter
4. If results seem off, try a different image angle

## Troubleshooting

Common issues and solutions:

1. **Application won't start**
   - Ensure all dependencies are installed
   - Check for Python version compatibility
   - Verify you're running from the correct directory

2. **Image upload errors**
   - Ensure image is in JPG, JPEG, or PNG format
   - Check that file size is under 200MB
   - Try a different image if problems persist

3. **Low confidence predictions**
   - Try a clearer image with less background
   - Use images that show distinctive features of the dinosaur
   - Consider that some species have similar appearances

4. **Slow performance**
   - Model inference requires sufficient computing resources
   - Close other resource-intensive applications
   - First prediction may take longer as the model loads

For further assistance, please open an issue on the project repository.
