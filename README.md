# Clinical EKG Arrhythmia Classifier

Welcome to the Clinical EKG Classifier project. This repository contains a production-grade PyTorch pipeline for multi-label classification of 12-lead EKG waveforms using the PhysioNet WFDB dataset. The model predicts up to 55 unique SNOMED-CT cardiovascular conditions.

## Dataset & Classifications

This project utilizes the **PhysioNet Large Scale 12-lead Electrocardiogram Database for Arrhythmia Study** (Zheng et al., 2022).

### The Data
- **Scale**: The dataset contains resting 12-lead EKGs from 45,152 distinct patients.
- **Format**: Data is provided in the standard WFDB format. Each record consists of a `.mat` file containing the raw binary signal and a `.hea` header file containing patient metadata and diagnosis annotations.
- **Signal Specs**: The waveforms are captured at a 500 Hz sampling rate, with voltages recorded in microvolts at a 32-bit A/D resolution.

### The Classifications (SNOMED-CT)
The diagnostic targets are **55 unique cardiovascular conditions** encoded using the standardized SNOMED-CT clinical vocabulary. 
This is treated as a **Multi-Label Classification** problem, as a single patient's 60-second EKG recording can simultaneously exhibit multiple conditions (e.g., Atrial Fibrillation *and* a Right Bundle Branch Block).

To map the diagnostic text from the header files to the 55 multi-hot target tensors, the dataset uses the `ConditionNames_SNOMED-CT.csv` mapping dictionary.

## Model Architecture (`WFDBResNetLSTM`)

Our model is an advanced Convolutional Recurrent Neural Network (CRNN) that utilizes Squeeze-and-Excitation mechanisms and sequence-level Self-Attention to maximize performance and interpretability.

### 1. Convolutional Stem

The raw 12-lead EKG signal (sampled at 500Hz) first passes through a downsampling convolutional stem. Because EKG waveforms are high-frequency, a large `kernel_size=15` and `stride=3` are used alongside MaxPooling to quickly compress the temporal dimension and extract foundational low-level features.

### 2. Squeeze-and-Excitation (SE) ResNet Blocks

The core feature extractor relies on 1D Residual Networks (`BasicBlock1d`). Inside each residual block, we've integrated a **Squeeze-and-Excitation (SE)** module.

- The SE module uses Adaptive Average Pooling to evaluate the global context of the waveform.
- It calculates dynamic, learnable weights for each of the 12 channels (EKG leads). 
- This allows the network to suppress noisy leads and amplify clinically relevant leads on a per-patient basis.

### 3. Bidirectional LSTM

Because EKG morphology evolves over time, the ResNet features are piped into a 2-layer Bidirectional LSTM. The LSTM learns the sequential dependencies of the QRS complexes across the entire variable-length 60-second window.

### 4. Self-Attention Mechanism

Instead of taking the naive final hidden state of the LSTM, we utilize a **Self-Attention Layer**.

- A feed-forward network calculates an attention score for every timestep.
- A `softmax` function normalizes these scores into a probability distribution.
- The LSTM outputs are multiplied by these attention weights to create a final **Context Vector**.
- This enables the model to explicitly ignore segments of baseline wander or muscle artifact, and strongly focus on segments containing the actual arrhythmia.

## Data Pipeline & Augmentation

The pipeline is built for clinical robustness:

- **Variable-Length Inputs**: Using a custom `pad_collate` function in the PyTorch `DataLoader`, the model dynamically supports EKG waveforms ranging anywhere from 10 seconds to over 60 seconds long without hardcoding limits.
- **Random Gaussian Noise**: Simulates baseline wander and sensor noise to prevent overfitting.
- **Lead Masking**: Randomly zeroes out an entire lead during training, forcing the SE network to learn redundant feature representations across the remaining 11 leads.

## Training & Deployment (Databricks / MLflow)

The model is natively integrated with MLflow for enterprise tracking and deployment.

- **Loss Function**: `FocalLoss` is utilized to heavily penalize mistakes on rare conditions (like acute myocardial infarctions) in the highly imbalanced dataset.
- **Metrics**: The training loop calculates `Macro F1`, `Macro ROC-AUC`, and `Macro PR-AUC` for rigorous clinical evaluation.
- **Threshold Calibration**: The system sweeps thresholds from 0.10 to 0.90 after training to automatically calculate the mathematically optimal threshold for maximum Macro F1 sensitivity.
- **Databricks Serving**: Upon completion, the model infers its exact input/output tensor signature and runs `mlflow.pytorch.log_model()`. The model is immediately registered to the MLflow database and is 100% ready for deployment as a Databricks REST API Model Serving endpoint.

## Usage

**Train the Model & Log to MLflow:**
```bash
python train.py
```

**Run Inference on Sample Patients:**
```bash
python predict.py
```

## References

When using or referencing this project, please acknowledge the original dataset authors and the PhysioNet resource:

1. Zheng, J., Guo, H., & Chu, H. (2022). A large scale 12-lead electrocardiogram database for arrhythmia study (version 1.0.0). *PhysioNet*. [https://doi.org/10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52)
2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation [Online]*. 101 (23), pp. e215–e220.
