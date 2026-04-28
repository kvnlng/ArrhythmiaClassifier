import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from pyspark.sql import SparkSession
from torch.utils.data import Dataset, DataLoader

# Import our custom architecture and loss function
from models.wfdb_resnet import WFDBResNetLSTM
from models.loss import FocalLoss


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


def finetune():
    # 1. Initialize Spark Session (automatically available in Databricks)
    spark = SparkSession.builder.getOrCreate()

    print("Loading data from Delta Table...")
    # Read the 60-second clips from the Delta Table
    # NOTE: Replace 'catalog.schema.ekg_60s_clips' with your actual Unity Catalog path
    df = spark.table("catalog.schema.ekg_60s_clips")

    # 2. Convert Spark DataFrame to Pandas DataFrame
    # This brings the data into the driver node's RAM for rapid PyTorch iteration
    pdf = df.toPandas()
    print(f"Loaded {len(pdf)} 60-second EKG clips into memory.")

    # Initialize PyTorch Dataset and DataLoader
    dataset = DeltaEKGDataset(pdf, num_classes=55)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Setup Device (Databricks GPU clusters usually map to cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Initialize Model and Load Pre-trained Weights
    model = WFDBResNetLSTM(num_classes=55)

    # Load the weights from our V1 training
    # If the script is run in Databricks Repos, this path will resolve correctly
    state_dict = torch.load("best_wfdb_resnet_lstm.pth", map_location=device)
    model.load_state_dict(state_dict)

    # Freeze the Convolutional Stem to retain general EKG feature extraction
    print("Freezing the Convolutional Stem...")
    for param in model.stem.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 5. Define Loss and Optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Use a smaller learning rate for fine-tuning, and only pass the unfrozen parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    num_epochs = 10

    # 6. Set MLflow Experiment
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
        # Create a dummy input representing a 12-lead, 60s @ 250Hz sample
        sample_input = np.random.randn(1, 12, 15000).astype(np.float32)
        signature = mlflow.models.signature.infer_signature(
            sample_input,
            model(torch.tensor(sample_input).to(device)).detach().cpu().numpy(),
        )

        mlflow.pytorch.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name="EKG_Classifier_Finetuned",
        )
        print("Model successfully registered to MLflow as EKG_Classifier_Finetuned!")


if __name__ == "__main__":
    finetune()
