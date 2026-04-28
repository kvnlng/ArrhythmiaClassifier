import os
import random
import torch
from data.wfdb_dataset import WFDBDataset
from models.wfdb_resnet import WFDBResNetLSTM


def main():
    # 1. Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # 2. Load dataset (to get the SNOMED-CT mappings)
    # We only need the mappings, but we can also use it to grab a random patient
    base_dir = os.path.join(os.path.dirname(__file__), "data", "wfdb")
    print("Loading dataset mappings...")
    dataset = WFDBDataset(base_dir)

    # 3. Load the best trained model
    best_model_path = os.path.join(
        os.path.dirname(__file__), "best_wfdb_resnet_lstm.pth"
    )
    if not os.path.exists(best_model_path):
        print(f"Error: Could not find model weights at {best_model_path}")
        return

    model = WFDBResNetLSTM(num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Loaded best_wfdb_resnet_lstm.pth successfully!\n")

    # 4. Grab a random patient record
    idx = random.randint(0, len(dataset) - 1)
    record_path = dataset.records[idx]
    patient_id = os.path.basename(record_path)
    print(f"=== Patient Record: {patient_id} ===")

    # Get the raw tensor and the actual multi-hot labels
    ekg_tensor, actual_labels = dataset[idx]

    # Reverse lookup for labels
    idx_to_code = {v: k for k, v in dataset.code_to_idx.items()}

    # 5. Determine ACTUAL diseases
    actual_indices = (actual_labels == 1.0).nonzero(as_tuple=True)[0]
    actual_diseases = [idx_to_code[i.item()] for i in actual_indices]

    print("\n[ ACTUAL DIAGNOSES (Ground Truth) ]")
    if len(actual_diseases) == 0:
        print("  - None (Normal / No SNOMED code found)")
    else:
        for d in actual_diseases:
            print(f"  - SNOMED-CT Code: {d}")

    # 6. Run Inference (PREDICTED diseases)
    # PyTorch models expect batches, so we add a batch dimension of 1
    # Shape could be anything up to (1, 12, 30000) for a 60 second EKG!
    input_tensor = ekg_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        # Apply sigmoid to convert raw logits to probabilities (0.0 to 1.0)
        probabilities = torch.sigmoid(logits)[0]

    # Read the calibrated threshold from train.py if it exists
    thresh_path = os.path.join(os.path.dirname(__file__), "optimal_threshold.txt")
    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            threshold = float(f.read().strip())

    print(
        f"\n[ PREDICTED DIAGNOSES (Neural Network) ] - Calibrated Threshold: {threshold:.2f}"
    )
    # We predict positive if probability > calibrated threshold
    predicted_indices = (probabilities > threshold).nonzero(as_tuple=True)[0]

    if len(predicted_indices) == 0:
        print(f"  - None (Normal / No disease detected >{threshold * 100:.0f}%)")
    else:
        for i in predicted_indices:
            idx_val = i.item()
            code = idx_to_code[idx_val]
            prob = probabilities[idx_val].item() * 100
            print(f"  - SNOMED-CT Code: {code} (Confidence: {prob:.2f}%)")

    print("\n[ TOP 3 HIGHEST PROBABILITIES ]")
    # Even if they aren't > 50%, what was the model leaning towards?
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    for prob, idx in zip(top3_probs, top3_indices):
        code = idx_to_code[idx.item()]
        print(f"  - {code}: {prob.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
