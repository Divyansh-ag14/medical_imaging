# Pneumonia Detection from Chest X-rays with Deep Learning and Class Activation Maps (CAM)

## Project Overview

This project implements a deep learning pipeline to detect pneumonia in chest X-ray images. It uses a modified ResNet18 CNN architecture, trained on preprocessed grayscale images. In addition to classification, the project implements **Class Activation Maps (CAM)** to provide visual interpretability, highlighting image regions most relevant to the model’s prediction. This improves transparency and trust in the AI model’s decisions.

---

## Data Preparation

- Input images are preprocessed grayscale chest X-rays saved as `.npy` NumPy arrays.
- Images have a fixed size of **224x224** pixels.
- The dataset is divided into training and validation folders: `Processed/train/` and `Processed/val/`.
- Images are loaded using a custom loader function `load_file(path)` which reads `.npy` files as float32 arrays.
- Data transformations apply:
  - Conversion to PyTorch tensors (`ToTensor()`)
  - Normalization with mean `0.49` and standard deviation `0.248` to center pixel values, helping model convergence.

---

## Model Architecture and Implementation

The backbone of the model is a **ResNet18** CNN pre-trained on ImageNet, adapted as follows:

- **Input Layer:** Modified the first convolutional layer (`conv1`) to accept **1 input channel** (grayscale) instead of 3 (RGB).
- **Output Layer:** Replaced the final fully connected (`fc`) layer to output a single neuron for binary classification (pneumonia vs. no pneumonia).
- Extracted the last convolutional feature maps (before pooling and fc layer) as a separate module (`self.feature_map`), enabling CAM computation.
- Forward pass details:
  - Extract feature maps from the last conv layer.
  - Perform **adaptive average pooling** to reduce the feature maps from `512 x 7 x 7` to `512 x 1 x 1`.
  - Flatten pooled features to a vector of length 512.
  - Pass through the fully connected layer to obtain the final logit output.
  - Return both the logit and the feature map tensor for interpretability.

---

## Training Pipeline (from `1_train_model.ipynb`)

- Uses **PyTorch Lightning** for modular and clean training.
- Optimizer: Adam with learning rate scheduling.
- Loss Function: `BCEWithLogitsLoss` which combines sigmoid activation and binary cross-entropy.
- Batch size, epochs, and other hyperparameters configured for efficient training on GPU.
- During training:
  - Compute predictions for batches.
  - Calculate loss and backpropagate.
  - Validate on the validation set and track metrics like accuracy.
- Model checkpoints saved for later evaluation and visualization.

---

## Evaluation Pipeline (from `2_evaluate_model.ipynb`)

- Loads the trained model weights.
- Runs inference on validation data.
- Converts logits to probabilities using sigmoid.
- Applies threshold (0.5) for classification decision.
- Calculates common classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Outputs these metrics to assess model performance.

---

## Class Activation Maps (from `3_interpretability.ipynb`)

CAM allows identification of image regions most influential in the model’s decision:

- Extract the output of the last convolutional layer (`512 x 7 x 7`).
- Flatten the spatial dimensions to `(512, 49)` where 49 = 7 x 7.
- Extract the weights from the fully connected layer (`fc`), corresponding to the classification neuron.
- Compute the weighted sum over the feature maps:
  
  \[
  M = \sum_{k} w_k A_k
  \]
  
  where \(w_k\) is the weight for channel \(k\) and \(A_k\) is the feature map channel \(k\).
- Normalize the resulting activation map between 0 and 1.
- Resize the 7x7 CAM to the original image size (224x224) using interpolation.
- Overlay the CAM heatmap on the original image using a color map (jet) with transparency.
- Display both the original image and the CAM overlay side-by-side to visually interpret which parts of the lung X-ray influenced the pneumonia prediction.

---

## Usage Instructions

1. **Prepare Data:** Preprocess your chest X-ray images as grayscale `.npy` arrays resized to 224x224 pixels and organize into train/validation folders.
2. **Train Model:** Run `1_train_model.ipynb` to train the CNN on your data.
3. **Evaluate Model:** Use `2_evaluate_model.ipynb` to load the trained model and calculate classification metrics.
4. **Visualize Predictions:** Use `3_interpretability.ipynb` to generate and visualize Class Activation Maps for individual X-ray images.

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- PyTorch Lightning
- numpy
- matplotlib
- scikit-learn

---

## Summary

This project provides a full pipeline for:

- Preprocessing medical images for CNN input
- Training a ResNet18-based classifier on grayscale X-ray data
- Evaluating classification performance with standard metrics
- Applying CAMs to interpret and visualize model decisions on a spatial level

This combination of automated detection and interpretability makes the system useful for research, clinical decision support, and educational purposes.



