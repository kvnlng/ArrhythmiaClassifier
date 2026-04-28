import os
import torch
import math


def generate_ekg_dataset(base_dir, num_train=100, num_val=20):
    splits = {"train": num_train, "val": num_val}
    classes = ["sinus_rhythm", "vt"]

    # 1 minute at 250 Hz
    num_samples = 60 * 250
    num_leads = 12

    # Create time array (seconds)
    t = torch.linspace(0, 60, num_samples)

    for split, num_files in splits.items():
        for cls_name in classes:
            dir_path = os.path.join(base_dir, split, cls_name)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(num_files):
                # We'll create a tensor of shape (12, 15000)
                ekg = torch.zeros(num_leads, num_samples)

                if cls_name == "sinus_rhythm":
                    # Normal heart rate: ~60-100 bpm (1-1.6 Hz)
                    # We'll simulate a simple 1.2 Hz wave + some noise
                    hz = 1.2 + (torch.rand(1).item() - 0.5) * 0.4
                    base_wave = torch.sin(2 * math.pi * hz * t)

                    for lead in range(num_leads):
                        # Each lead sees the wave slightly differently
                        amplitude = 0.5 + torch.rand(1).item()
                        noise = torch.randn(num_samples) * 0.1
                        ekg[lead] = amplitude * base_wave + noise

                elif cls_name == "vt":
                    # Ventricular Tachycardia: Fast, wide complexes ~150-250 bpm (2.5-4 Hz)
                    hz = 3.0 + (torch.rand(1).item() - 0.5) * 1.0
                    # VT has a different, often more sinusoidal or jagged shape, we'll use a fast sine
                    base_wave = torch.sin(2 * math.pi * hz * t)

                    for lead in range(num_leads):
                        amplitude = (
                            1.0 + torch.rand(1).item()
                        )  # Usually higher amplitude
                        noise = torch.randn(num_samples) * 0.2
                        ekg[lead] = amplitude * base_wave + noise

                # Save the tensor directly
                file_path = os.path.join(dir_path, f"{i:03d}.pt")
                torch.save(ekg, file_path)

    print(f"EKG Dataset generated at {base_dir}")
    print(f"Train: {num_train} samples per class")
    print(f"Val: {num_val} samples per class")


if __name__ == "__main__":
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "ekg_data"
    )
    generate_ekg_dataset(base_dir)
