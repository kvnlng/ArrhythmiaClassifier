# Databricks notebook source
# MAGIC %md
# MAGIC # EKG Arrhythmia Classifier - Databricks Fine-Tuning
# MAGIC This notebook demonstrates how to load the pre-trained `WFDBResNetLSTM` model and fine-tune it
# MAGIC on a new dataset stored in a Unity Catalog Delta Table.

# COMMAND ----------

# MAGIC %pip install torch torchvision mlflow numpy pandas pyspark wfdb

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Imports & Setup

# COMMAND ----------

import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from pyspark.sql import SparkSession
from torch.utils.data import Dataset, DataLoader

# Ensure the repo root is in the path to import custom modules
sys.path.append(os.path.abspath('.'))

# Import our custom architecture and loss function
from models.wfdb_resnet import WFDBResNetLSTM
from models.loss import FocalLoss

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Define Dataset for Delta Tables

# COMMAND ----------

class DeltaEKGDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps a Pandas DataFrame extracted from a Delta Table.
    """
    def __init__(self, pdf: pd.DataFrame, num_classes: int = 55):
        """
        Assumes the DataFrame contains:
        - 'waveform_array': A list of lists or 2D numpy array of shape (12, 15000)
        - 'labels': A list or 1D numpy array of shape (55,) indicating the multi-hot SNOMED-CT codes
        """
        self.waveforms = pdf["waveform_array"].values
        self.labels = pdf["labels"].values
        self.num_classes = num_classes

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # Convert the array to float32 tensors
        # Shape should be (12, 15000) for 12-lead EKG at 250Hz for 60 seconds
        x = np.array(self.waveforms[idx], dtype=np.float32)
        y = np.array(self.labels[idx], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load Data from Unity Catalog
# MAGIC We read the 60-second EKG clips directly into the driver's memory as a Pandas DataFrame for rapid PyTorch iteration.

# COMMAND ----------

# Initialize Spark Session (automatically available in Databricks)
spark = SparkSession.builder.getOrCreate()

print("Loading data from Delta Table...")
# NOTE: Replace 'catalog.schema.ekg_60s_clips' with your actual Unity Catalog path
df = spark.table("catalog.schema.ekg_60s_clips")

# Convert Spark DataFrame to Pandas DataFrame
pdf = df.toPandas()
print(f"Loaded {len(pdf)} 60-second EKG clips into memory.")

# Initialize PyTorch Dataset and DataLoader
dataset = DeltaEKGDataset(pdf, num_classes=55)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Initialize Model and Load Pre-trained Weights

# COMMAND ----------

# Setup Device (Databricks GPU clusters usually map to cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = WFDBResNetLSTM(num_classes=55)

# Load the weights from our V1 training
# If the notebook is running in Databricks Repos, this path will resolve correctly
best_model_path = "best_wfdb_resnet_lstm.pth"
if os.path.exists(best_model_path):
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded pre-trained weights successfully!")
else:
    print(f"Warning: Could not find '{best_model_path}'. Initializing with random weights.")

# Freeze the Convolutional Stem to retain general EKG feature extraction
print("Freezing the Convolutional Stem...")
for param in model.stem.parameters():
    param.requires_grad = False

model = model.to(device)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Define Loss and Optimizer

# COMMAND ----------

criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Use a smaller learning rate for fine-tuning, and only pass the unfrozen parameters
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6. Fine-Tune with MLflow Tracking

# COMMAND ----------

num_epochs = 10

# Set MLflow Experiment
mlflow.set_experiment("/Shared/ekg_finetuning")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-5)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("frozen_stem", True)

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

    print("Fine-tuning complete!")

    # 7. Log the fine-tuned model to the MLflow Registry
    print("\nLogging fine-tuned model to MLflow...")
    
    # Create a dummy input representing a 12-lead, 60s @ 250Hz sample
    sample_input = np.random.randn(1, 12, 15000).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        sample_output = model(torch.tensor(sample_input).to(device)).cpu().numpy()
        
    signature = mlflow.models.signature.infer_signature(
        sample_input,
        sample_output,
    )

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="ekg_finetuned_model",
        signature=signature,
        input_example=sample_input,
        serialization_format="pt2",
        pip_requirements="requirements.txt",
        registered_model_name="EKG_Classifier_Finetuned",
    )
    print("Model successfully registered to MLflow as EKG_Classifier_Finetuned!")
