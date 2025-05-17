# Deploying to Streamlit

This document describes how to deploy this Streamlit application with the model files correctly configured.

## Creating GitHub Releases for Model Files and Dataset

Since the `model/` directory and some data files are excluded from git tracking (in `.gitignore`), we need to make these files available for the Streamlit app deployment through GitHub releases.

### Model Files

1. Compress your `model/` directory into a ZIP file:

   ```bash
   # Windows (PowerShell)
   Compress-Archive -Path .\model\* -DestinationPath model.zip
   
   # Linux/Mac
   zip -r model.zip model/
   ```

2. Create a GitHub release:
   - Go to your GitHub repository
   - Click on "Releases" in the right sidebar
   - Click "Create a new release" or "Draft a new release"
   - Set a tag (e.g., "v1.0")
   - Add a title and description
   - Upload the `model.zip` file
   - Publish the release

3. Update the URL in your code:
   - In `PaleoNet.py`, update the `github_model_url` variable with the actual URL to your release ZIP file
   - The URL should look something like: `https://github.com/SimonStnn/PaleoNet/releases/download/v0.0.0-pre/model.zip`

### Dataset Files

1. Compress your dataset directory into a ZIP file:

   ```bash
   # Windows (PowerShell)
   Compress-Archive -Path .\data\dinosaur_dataset_split -DestinationPath dataset.zip
   
   # Linux/Mac
   zip -r dataset.zip data/dinosaur_dataset_split
   ```

2. Add the dataset ZIP to the same release:
   - Go to your existing release or create a new one
   - Upload the `dataset.zip` file
   - Update the release

3. Update the URL in your code:
   - In `PaleoNet.py`, update the `github_dataset_url` variable with the actual URL to your dataset ZIP file
   - The URL should look something like: `https://github.com/SimonStnn/PaleoNet/releases/download/v0.0.0-pre/dataset.zip`

## Deploying to Streamlit Cloud

1. Ensure your repository has:
   - All code files (excluding the `model/` and `data/` directories)
   - `requirements.txt` file (including all dependencies)
   - A GitHub release with the model and dataset files

2. Connect your GitHub repository to Streamlit Cloud:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and main file (`PaleoNet.py`)
   - Deploy the app

The app will automatically download the model files from your GitHub release when it starts up if they're not already present.

## Testing Locally

To test if your implementation works correctly:

1. Delete your local `model/` and `data/dinosaur_dataset_split` directories
2. Run the app: `streamlit run PaleoNet.py`
3. The app should automatically download both the model files and dataset from your GitHub releases

## Troubleshooting

If you encounter any issues:

- Check if the GitHub release URLs are correct
- Ensure the model and dataset files are properly zipped with the correct directory structure
- Check the Streamlit logs for any error messages
- Verify that all required packages are in your `requirements.txt` file
- The app can run with just the model files if the dataset can't be downloaded, but some features will be limited
