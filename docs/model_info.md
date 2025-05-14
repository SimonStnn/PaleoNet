# PaleoNet Model Information

This document provides detailed information about the deep learning model used in PaleoNet for dinosaur species classification.

## Model Architecture

PaleoNet uses a transfer learning approach based on the EfficientNetB0 architecture:

1. **Base Model**: EfficientNetB0 pre-trained on ImageNet dataset
   - EfficientNet is a family of models that efficiently scales width, depth, and resolution for better performance
   - EfficientNetB0 is the smallest variant, offering a good balance between accuracy and computational efficiency

2. **Custom Classification Head**:
   - Global Average Pooling
   - Dense layer with 256 units and ReLU activation
   - Dropout (0.5) for regularization
   - Final Dense layer with 15 units (one per dinosaur species) and softmax activation

## Model Training

The model was trained using the following approach:

1. **Transfer Learning Strategy**:
   - Frozen base model (pre-trained weights)
   - Training only the custom classification head initially
   - Fine-tuning the top layers of the base model in later stages

2. **Training Parameters**:
   - Optimizer: Adam with learning rate 0.001
   - Loss function: Categorical Cross-entropy
   - Batch size: 32
   - Epochs: 30 (with early stopping)

3. **Data Augmentation**:
   - Random rotation (±20°)
   - Random zoom (±20%)
   - Random horizontal flip
   - Random brightness adjustment
   - Random contrast adjustment

## Dataset

The model was trained on a dataset of dinosaur images:

- **Total dataset size**: ~3,000 images
- **Classes**: 15 different dinosaur species
- **Split**: 70% training, 15% validation, 15% testing
- **Image dimensions**: Resized to 296×296 pixels, RGB

## Performance Metrics

The model achieves the following performance on the test dataset:

- **Overall Accuracy**: 80.9%
- **Average Precision**: 81.5%
- **Average Recall**: 81.0%
- **Average F1-Score**: 80.7%

### Per-Class Performance

| Species            | Precision | Recall | F1-Score |
| ------------------ | --------- | ------ | -------- |
| Ankylosaurus       | 87.2%     | 84.0%  | 85.6%    |
| Brachiosaurus      | 73.1%     | 76.5%  | 74.8%    |
| Compsognathus      | 68.4%     | 65.0%  | 66.7%    |
| Corythosaurus      | 71.2%     | 74.5%  | 72.8%    |
| Dilophosaurus      | 79.3%     | 76.2%  | 77.7%    |
| Dimorphodon        | 85.6%     | 82.1%  | 83.8%    |
| Gallimimus         | 64.5%     | 60.2%  | 62.3%    |
| Microceratus       | 72.3%     | 68.0%  | 70.1%    |
| Pachycephalosaurus | 80.1%     | 83.5%  | 81.8%    |
| Parasaurolophus    | 81.6%     | 79.0%  | 80.3%    |
| Spinosaurus        | 84.2%     | 87.5%  | 85.8%    |
| Stegosaurus        | 89.0%     | 88.2%  | 88.6%    |
| Triceratops        | 90.5%     | 89.7%  | 90.1%    |
| Tyrannosaurus Rex  | 91.2%     | 92.0%  | 91.6%    |
| Velociraptor       | 73.6%     | 70.5%  | 72.0%    |

## Confusion Matrix Analysis

The model performs particularly well on species with distinctive features, such as:

- Triceratops (distinct head shield and horns)
- Tyrannosaurus Rex (large head, small arms)
- Stegosaurus (distinctive plates)

The model sometimes confuses visually similar species:

- Gallimimus and Velociraptor (both bipedal theropods)
- Corythosaurus and Parasaurolophus (both hadrosaurids with crests)
- Compsognathus and Microceratus (both small-bodied dinosaurs)

## Model Limitations

1. **Image Quality Dependency**: Performance is better with clear, well-lit images
2. **Pose Sensitivity**: Some species are more recognizable from specific angles
3. **Background Influence**: Complex backgrounds can sometimes affect classification
4. **Artistic Variation**: Different artistic interpretations of dinosaurs can lead to inconsistencies

## Future Improvements

Potential areas for model enhancement:

1. **Larger Dataset**: Collecting more diverse images per species
2. **Deeper Architecture**: Testing larger EfficientNet variants (B1-B7)
3. **Additional Augmentation**: More aggressive augmentation to improve generalization
4. **Ensemble Approach**: Combining multiple models for better accuracy
5. **Attention Mechanisms**: Incorporating attention to focus on distinctive features
