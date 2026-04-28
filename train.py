import torch
import torch.optim as optim

# Import our custom models
from models.wfdb_resnet import WFDBResNetLSTM
from models.loss import FocalLoss
from data.wfdb_dataset import WFDBDataset
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature


def pad_collate(batch):
    """
    Custom collate_fn to pad variable-length 1D ECG tensors.
    """
    tensors = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Transpose for pad_sequence: (channels, seq_len) -> (seq_len, channels)
    tensors_transposed = [t.transpose(0, 1) for t in tensors]

    # Pad them! (Finds the longest in the batch and pads the rest with 0)
    padded_tensors = pad_sequence(
        tensors_transposed, batch_first=True, padding_value=0.0
    )

    # Transpose back: (batch, max_seq_len, channels) -> (batch, channels, max_seq_len)
    padded_tensors = padded_tensors.transpose(1, 2)

    # Stack labels
    labels = torch.stack(labels)

    return padded_tensors, labels


def train():
    # 1. Setup device (use GPU if available)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Setup Real DataLoaders
    base_dir = os.path.join(os.path.dirname(__file__), "data", "wfdb")

    # Use is_train=True for training to enable Data Augmentation!
    full_dataset_train = WFDBDataset(base_dir, is_train=True)
    full_dataset_val = WFDBDataset(base_dir, is_train=False)

    # We need to split the dataset into train and val
    train_size = int(0.8 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size

    # We must use the exact same generator seed to ensure train/val splits are identical for both dataset objects
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = torch.utils.data.random_split(
        full_dataset_train, [train_size, val_size], generator=generator
    )
    _, val_dataset = torch.utils.data.random_split(
        full_dataset_val, [train_size, val_size], generator=generator
    )

    # Batch size is larger for big data to speed up training
    # num_workers=4 enables multi-processing to read files in parallel
    # We must pass collate_fn=pad_collate because we now have variable length tensors!
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate,
        persistent_workers=True,
    )

    # 3. Initialize Model
    num_classes = full_dataset_train.num_classes
    model = WFDBResNetLSTM(num_classes=num_classes)

    model = model.to(device)

    # 3. Define Loss function and Optimizer
    # We removed extreme pos_weights because Focal Loss inherently handles imbalance using alpha
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Adam is a popular, robust optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 50
    # Learning Rate Scheduler: CosineAnnealingLR gracefully decays LR to 1e-6 over 50 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(
        os.path.dirname(__file__), "best_wfdb_resnet_lstm.pth"
    )

    # Start MLflow run for Databricks tracking
    mlflow.set_experiment("/Shared/ekg_classifier")
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model_type", "wfdb_resnet_lstm")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "CosineAnnealingLR")

        # Early stopping parameters
        patience = 3
        epochs_no_improve = 0

        print(f"\nStarting training WFDB_RESNET_LSTM for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0

            # Wrap train_loader with tqdm for a progress bar
            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
            )
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                # Move data to the same device as the model
                inputs, targets = inputs.to(device), targets.to(device)

                # Step 1: Zero the gradients (clear out gradients from previous step)
                optimizer.zero_grad()

                # Step 2: Forward pass (compute predictions)
                outputs = model(inputs)

                # For multi-label, predictions are positive if logit > 0 (prob > 0.5)
                predicted = (outputs.data > 0.0).float()
                # Count how many individual labels across all samples and classes were predicted correctly
                correct += (predicted == targets).sum().item()

                # Step 3: Compute loss
                loss = criterion(outputs, targets)

                # Step 4: Backward pass (compute gradients for each parameter)
                loss.backward()

                # LSTMs suffer from exploding gradients. We clip them to a max_norm of 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step 5: Update weights using the optimizer
                optimizer.step()

                running_loss += loss.item()

                # Update the progress bar description with the current batch loss
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Print average loss for the epoch
            avg_loss = running_loss / len(train_loader)
            # Accuracy is computed over total individual label predictions
            total_labels = len(train_loader.dataset) * num_classes
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
            print(
                f"Training Accuracy (Multi-Label): {100 * correct / total_labels:.2f}%"
            )

            # --- VALIDATION PHASE ---
            model.eval()  # Tell the model we are evaluating, not training
            val_loss = 0.0
            val_correct = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            with torch.no_grad():
                for val_inputs, val_targets in val_pbar:
                    # 1. Move val_inputs and val_targets to the 'device'
                    val_inputs, val_targets = (
                        val_inputs.to(device),
                        val_targets.to(device),
                    )

                    # 2. Get predictions: outputs = model(val_inputs)
                    outputs = model(val_inputs)
                    # 3. Calculate loss: loss = criterion(outputs, val_targets)
                    loss = criterion(outputs, val_targets)

                    # 4. Find predicted classes
                    predicted = (outputs.data > 0.0).float()
                    val_correct += (predicted == val_targets).sum().item()

                    # 5. Add to val_loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            total_val_labels = len(val_loader.dataset) * num_classes
            avg_val_acc = 100 * val_correct / total_val_labels

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f} Validation Accuracy (Multi-Label): {avg_val_acc:.2f}%"
            )

            # Step the learning rate scheduler (CosineAnnealingLR steps every epoch)
            scheduler.step()

            # Checkpoint the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                print(
                    f"--> Validation loss improved! Saving model to {best_model_path}"
                )
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                print(
                    f"--> No improvement in validation loss for {epochs_no_improve} epoch(s)."
                )
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered! Training stopped after {epoch + 1} epochs."
                    )
                    break

        print("Training finished!")

        # 5. Threshold Calibration
        print("\n--- Performing Threshold Calibration on Best Model ---")
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        all_probs = []
        all_targets = []

        calib_pbar = tqdm(val_loader, desc="Calibrating Thresholds")
        with torch.no_grad():
            for val_inputs, val_targets in calib_pbar:
                val_inputs = val_inputs.to(device)
                outputs = model(val_inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(val_targets.numpy())

        all_probs = np.vstack(all_probs)
        all_targets = np.vstack(all_targets)

        best_threshold = 0.5
        best_f1 = 0.0

        print("\nEvaluating F1-Scores across thresholds:")
        for thresh in np.arange(0.10, 0.95, 0.05):
            preds = (all_probs > thresh).astype(float)
            # Calculate Macro F1
            score = f1_score(all_targets, preds, average="macro", zero_division=0)
            print(f"  Threshold {thresh:.2f} -> Macro F1: {score:.4f}")
            if score > best_f1:
                best_f1 = score
                best_threshold = thresh

        print(f"\nOPTIMAL THRESHOLD: {best_threshold:.2f} (Macro F1: {best_f1:.4f})")

        # Calculate ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average="macro")
            pr_auc = average_precision_score(all_targets, all_probs, average="macro")
            print(f"Macro ROC-AUC: {roc_auc:.4f}")
            print(f"Macro PR-AUC: {pr_auc:.4f}")

            # MLflow SQLite backend throws IntegrityError on NaNs due to a known bug
            if not np.isnan(roc_auc):
                mlflow.log_metric("final_roc_auc", float(roc_auc))
            if not np.isnan(pr_auc):
                mlflow.log_metric("final_pr_auc", float(pr_auc))

        except Exception as e:
            print(f"Could not calculate AUC: {e}")

        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("best_macro_f1", best_f1)

        # Save the optimal threshold to a text file for predict.py to use
        thresh_path = os.path.join(os.path.dirname(__file__), "optimal_threshold.txt")
        with open(thresh_path, "w") as f:
            f.write(str(best_threshold))
        print(f"Optimal threshold saved to {thresh_path}")

        # Log Model to MLflow with Signature for Databricks Serving
        print("\nLogging model to MLflow for Databricks Model Serving...")
        # Take a sample input and output to infer signature
        sample_input = next(iter(val_loader))[0][0:2].numpy()  # shape (2, 12, seq_len)

        # We need to run inference on CPU to get sample output
        model.to("cpu")
        sample_output = model(torch.FloatTensor(sample_input)).detach().numpy()

        signature = infer_signature(sample_input, sample_output)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="ekg_crnn_model",
            signature=signature,
            input_example=sample_input,
            serialization_format="pt2",
            pip_requirements="requirements.txt",
            registered_model_name="EKG_CRNN_Classifier",
        )
        print("Model successfully logged to MLflow with signature!")


if __name__ == "__main__":
    train()
