# Databricks notebook source
# MAGIC %md
# MAGIC # EKG Arrhythmia Classifier - Databricks Inference
# MAGIC This notebook demonstrates how to load the registered PyTorch model from the MLflow Model Registry
# MAGIC and run inference on patient EKG data stored in a Unity Catalog Delta Table.

# COMMAND ----------

# MAGIC %pip install torch torchvision mlflow numpy pandas pyspark wfdb

# COMMAND ----------

import os
import sys
import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from pyspark.sql import SparkSession

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Setup MLflow and Load Model
# MAGIC We load the `EKG_CRNN_Classifier` directly from the MLflow Model Registry. 

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_name = "EKG_CRNN_Classifier"

# Get the latest version of the model
# For production, you could specify stages=["Production"] or alias based tracking
try:
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    model_uri = f"models:/{model_name}/{latest_version.version}"
    print(f"Loading model from: {model_uri}")
except Exception as e:
    print(f"Could not find model '{model_name}'. Make sure it was logged successfully in train.py.")
    raise e

# Load the PyTorch model from MLflow
model = mlflow.pytorch.load_model(model_uri)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded successfully to {device}!")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Fetch the Optimal Threshold
# MAGIC During training, we logged the `optimal_threshold` to MLflow. We can retrieve it dynamically
# MAGIC from the MLflow run associated with this model version.

# COMMAND ----------

try:
    run = client.get_run(latest_version.run_id)
    threshold = run.data.metrics.get("optimal_threshold", 0.5)
except Exception as e:
    print("Could not retrieve threshold from MLflow, defaulting to 0.5")
    threshold = 0.5

print(f"Using calibrated optimal threshold: {threshold:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load SNOMED-CT Mappings
# MAGIC We load the code mappings from the dataset to decode the predictions.
# MAGIC *(Assumes the notebook is running inside the Databricks Repo alongside the code/data)*

# COMMAND ----------

# Add the repo root to sys.path so we can import local modules
sys.path.append(os.path.abspath('.'))

from data.wfdb_dataset import WFDBDataset

base_dir = os.path.join(os.path.abspath('.'), "data", "wfdb")
try:
    dataset = WFDBDataset(base_dir)
    idx_to_code = {v: k for k, v in dataset.code_to_idx.items()}
    print(f"Successfully loaded {len(idx_to_code)} SNOMED-CT mappings.")
except Exception as e:
    print(f"Warning: Could not load WFDBDataset mapping from {base_dir}. Using generic indices.")
    idx_to_code = {i: f"Class_{i}" for i in range(55)}

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Load Sample Data from Unity Catalog
# MAGIC We read a small sample of EKG recordings from our Delta table.

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# NOTE: Replace with your actual catalog and schema
table_name = "catalog.schema.ekg_60s_clips"

print(f"Attempting to fetch a sample record from {table_name}...")
try:
    df_sample = spark.table(table_name).sample(fraction=0.1).limit(1)
    pdf = df_sample.toPandas()
    
    if len(pdf) == 0:
        print("No data found in the table.")
        raise ValueError("Empty table")
        
    sample_record = pdf.iloc[0]
    
    # Extract waveform and convert to tensor
    waveform_array = np.array(sample_record["waveform_array"], dtype=np.float32)
    
    # Ensure correct shape (12, seq_len)
    if len(waveform_array.shape) == 1:
        waveform_array = waveform_array.reshape(12, -1)
        
    # Add batch dimension: (1, 12, seq_len)
    input_tensor = torch.tensor(waveform_array).unsqueeze(0).to(device)
    print(f"Input tensor shape ready for inference: {input_tensor.shape}")

except Exception as e:
    print(f"Could not load from {table_name}. Falling back to a random local dataset sample if available...")
    if 'dataset' in locals():
        import random
        idx = random.randint(0, len(dataset) - 1)
        ekg_tensor, actual_labels = dataset[idx]
        input_tensor = ekg_tensor.unsqueeze(0).to(device)
        print(f"Loaded local sample. Input tensor shape: {input_tensor.shape}")
    else:
        print("No local dataset available. Please configure the Unity Catalog table name.")
        raise e

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Run Inference

# COMMAND ----------

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits)[0].cpu().numpy()

print(f"\n[ PREDICTED DIAGNOSES ] - Threshold: {threshold:.4f}")

predicted_indices = np.where(probabilities > threshold)[0]

if len(predicted_indices) == 0:
    print(f"  - None (Normal / No disease detected > {threshold * 100:.0f}%)")
else:
    for idx_val in predicted_indices:
        code = idx_to_code[idx_val]
        prob = probabilities[idx_val] * 100
        print(f"  - Code: {code} (Confidence: {prob:.2f}%)")

print("\n[ TOP 3 HIGHEST PROBABILITIES ]")
top3_indices = np.argsort(probabilities)[::-1][:3]
for idx in top3_indices:
    code = idx_to_code[idx]
    prob = probabilities[idx] * 100
    print(f"  - {code}: {prob:.2f}%")
