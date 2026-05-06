import os
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from models.wfdb_resnet import WFDBResNetLSTM
from data.cudb_dataset import CUDBDataset

warnings.filterwarnings("ignore", message=".*were assigned during export.*", category=UserWarning)


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    base_dir = os.path.join(os.path.dirname(__file__), "data", "cudb_data")

    print("Initializing datasets...")
    train_dataset = CUDBDataset(base_dir, segment_length=1250, is_train=True)
    val_dataset = CUDBDataset(base_dir, segment_length=1250, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    model = WFDBResNetLSTM(in_channels=1, num_classes=1).to(device)

    # Standard BCE Loss for binary classification
    # If the dataset is imbalanced (e.g. 20% positive), we can pass pos_weight
    pos_weight = torch.tensor([80.0 / 20.0]).to(device) # roughly 4:1 negative to positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_model_path = os.path.join(os.path.dirname(__file__), "best_cudb_resnet_lstm.pth")

    mlflow.set_experiment("/Shared/cudb_classifier")
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model_type", "cudb_resnet_lstm_1ch")

        patience = 5
        epochs_no_improve = 0

        print(f"\nStarting training CUDB binary classification for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
            for inputs, targets in train_pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            all_val_targets = []
            all_val_probs = []

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    outputs = model(val_inputs)
                    loss = criterion(outputs, val_targets)
                    
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    val_correct += (predicted == val_targets).sum().item()
                    val_total += val_targets.size(0)
                    
                    all_val_targets.extend(val_targets.cpu().numpy().flatten())
                    all_val_probs.extend(probs.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            val_targets_np = np.array(all_val_targets)
            val_preds_np = (np.array(all_val_probs) > 0.5).astype(float)
            val_f1 = f1_score(val_targets_np, val_preds_np, zero_division=0)

            print(f"Epoch [{epoch + 1}/{num_epochs}] Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

            scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                print(f"--> Validation loss improved! Saving model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break

        print("Training finished!")
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        
        all_targets = []
        all_probs = []
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                outputs = model(val_inputs.to(device))
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_targets.extend(val_targets.numpy().flatten())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Threshold calibration
        best_threshold = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > thresh).astype(float)
            score = f1_score(all_targets, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = thresh

        print(f"\nOPTIMAL THRESHOLD: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
            pr_auc = average_precision_score(all_targets, all_probs)
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
            mlflow.log_metric("final_roc_auc", float(roc_auc))
            mlflow.log_metric("final_pr_auc", float(pr_auc))
        except Exception as e:
            print(f"Could not calculate AUC: {e}")

        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)

        # Log model
        sample_input = next(iter(val_loader))[0][0:2].numpy()
        model.to("cpu")
        sample_output = model(torch.FloatTensor(sample_input)).detach().numpy()
        signature = infer_signature(sample_input, sample_output)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="cudb_crnn_model",
            signature=signature,
            input_example=sample_input,
            serialization_format="pt2",
            registered_model_name="CUDB_Malignant_Classifier",
        )
        print("Model logged successfully!")

if __name__ == "__main__":
    train()
