# PaleoNet Development Guide

This document provides information for developers interested in contributing to or extending the PaleoNet dinosaur classification project.

## Development Environment Setup

1. **Clone the Repository**

   ```powershell
   git clone https://github.com/yourusername/PaleoNet.git
   cd PaleoNet
   ```

2. **Create a Virtual Environment**

   ```powershell
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   .\venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

## Project Structure

```text
PaleoNet/
├── PaleoNet.py             # Main Streamlit application
├── utils.py                # Utility functions
├── pages/                  # Additional app pages
│   ├── 01_Model_Info.py
│   ├── 02_Dinosaur_Encyclopedia.py
│   └── 03_Model_Performance.py
├── assets/                 # Images and static assets
│   ├── banner.png
│   └── logo.png
├── data/                   # Dataset directory
│   └── dinosaur_dataset_split/
│       ├── train/          # Training data (70%)
│       ├── val/            # Validation data (15%)
│       └── test/           # Test data (15%)
├── docs/                   # Documentation
│   ├── development.md
│   ├── model_info.md
│   └── user_guide.md
├── model/                  # Saved model files
│   ├── best_model_checkpoint.h5
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── dinosaur_classifier_transfer_learning.keras
│   ├── dinosaur_class_mapping.json
│   ├── dinosaur_model_performance.json
│   ├── training_history.csv
│   ├── training_history.png
│   └── training_history_detailed.json
├── opdracht_CNN_stijnen_simon.ipynb  # Model training notebook
├── requirements.txt        # Project dependencies
├── README.md               # Main documentation
└── LICENSE                 # MIT License
```

## Key Components

### 1. Model Training (opdracht_CNN_stijnen_simon.ipynb)

The Jupyter notebook contains the complete workflow for:

- Loading and preprocessing the dataset
- Building the EfficientNetB0-based model
- Training with transfer learning
- Evaluating on test data
- Saving model artifacts

To retrain the model with different parameters or architectures, modify this notebook.

### 2. Streamlit Application (PaleoNet.py)

The main application file:

- Loads the trained model
- Provides the user interface with a tabbed navigation system
- Handles image upload and processing
- Displays classification results 
- Contains three main tabs: Home, Upload Image, and Sample Gallery

### 3. Pages (pages/)

Additional application pages:

- `01_Model_Info.py`: Displays model architecture and performance
- `02_Dinosaur_Encyclopedia.py`: Information about dinosaur species
- `03_Model_Performance.py`: Displays model performance metrics and visualizations

### 4. Utility Functions (utils.py)

Contains helper functions for:

- Image preprocessing
- Visualization
- Model interpretation

## Development Workflow

### Adding a New Feature

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**
   - Update application code
   - Add tests for your feature (if applicable)
   - Update documentation in `docs/`

3. **Run tests locally**

   ```powershell
   # If using pytest
   pytest tests/
   ```

4. **Create a pull request**
   - Provide a clear description of your changes
   - Reference any related issues

### Modifying the Model

To improve or change the classification model:

1. Open the training notebook `opdracht_CNN_stijnen_simon.ipynb`
2. Modify the model architecture, training parameters, or data augmentation
3. Retrain the model
4. Evaluate performance
5. Export the model artifacts:
   - `dinosaur_classifier_transfer_learning.keras`
   - `dinosaur_class_mapping.json`
   - `dinosaur_model_performance.json`
6. Place the new model artifacts in the `model/` directory

### Adding a New Page

To add a new page to the Streamlit application:

1. Create a new Python file in `pages/` (the filename should start with a number to control ordering)
2. Import needed modules, especially `streamlit as st`
3. Set page configuration at the top
4. Implement the page content
5. Update documentation to reference your new page

Example:

```python
# pages/03_Your_New_Page.py
import streamlit as st

st.set_page_config(
    page_title="PaleoNet - Your New Page",
    page_icon="🦖",
    layout="wide"
)

st.title("Your New Page Title")
st.markdown("## Your content here")

# Rest of your page implementation
```

## Path Handling Guidelines

The application uses robust path handling to ensure portability across different operating systems and environments. When working with file paths:

1. **Always use OS-independent path construction**

   ```python
   import os
   from pathlib import Path
   
   # Get the current file's directory
   current_dir = Path(__file__).parent
   
   # Navigate to parent directory
   root_dir = current_dir.parent
   
   # Create path to a file
   file_path = os.path.join(root_dir, "model", "model_file.keras")
   ```

2. **Avoid hardcoded relative paths**
   - Don't use: `"../model/file.json"`
   - Instead use: `os.path.join(root_dir, "model", "file.json")`

3. **Add error handling for file operations**

   ```python
   try:
       with open(file_path, "r") as f:
           data = json.load(f)
   except FileNotFoundError:
       st.error(f"Could not find file: {file_path}")
       # Provide fallback behavior
   ```

## Documentation

When adding features or making changes:

1. Update relevant documentation in `docs/`
2. Add inline comments for complex code sections
3. Update README.md if needed
4. Include example usage where appropriate

## Deployment

To deploy the application:

1. Ensure all dependencies are in `requirements.txt`
2. For Streamlit Cloud:
   - Push to GitHub
   - Connect repository to Streamlit Cloud
   - Configure settings as needed

3. For self-hosting:
   - Install dependencies
   - Run with `streamlit run PaleoNet.py`
   - Consider using Docker for containerization

## Getting Help

If you need assistance with development:

- Check existing documentation
- Look for similar issues in the issue tracker
- Contact the maintainers
- Create a new issue with a clear description of your problem
