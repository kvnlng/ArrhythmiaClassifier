import os
import requests


def download_file(url, out_path):
    if not os.path.exists(out_path):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Failed to download {url}: Status {r.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


def main():
    # We will download 100 records from the PhysioNet dataset
    # URL: https://physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/01/010/

    dl_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "physionet_data"
    )
    os.makedirs(dl_dir, exist_ok=True)

    base_url = "https://physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/01/010"

    print(f"Downloading 100 PhysioNet WFDB records to {dl_dir}...")

    for i in range(1, 101):
        rec_id = f"JS{i:05d}"

        # Download the header file (.hea)
        download_file(f"{base_url}/{rec_id}.hea", os.path.join(dl_dir, f"{rec_id}.hea"))

        # Download the raw binary data file (.mat)
        download_file(f"{base_url}/{rec_id}.mat", os.path.join(dl_dir, f"{rec_id}.mat"))

        if i % 10 == 0:
            print(f"Downloaded {i} records...")

    print("Download complete!")


if __name__ == "__main__":
    main()
