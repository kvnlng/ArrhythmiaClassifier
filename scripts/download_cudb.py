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
    # We will download the 35 records from the CUDB PhysioNet dataset
    # URL: https://physionet.org/files/cudb/1.0.0/

    dl_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "cudb_data"
    )
    os.makedirs(dl_dir, exist_ok=True)

    base_url = "https://physionet.org/files/cudb/1.0.0"

    print(f"Downloading 35 CUDB records to {dl_dir}...")

    for i in range(1, 36):
        rec_id = f"cu{i:02d}"

        # Download the header file (.hea)
        download_file(f"{base_url}/{rec_id}.hea", os.path.join(dl_dir, f"{rec_id}.hea"))

        # Download the raw binary data file (.dat)
        download_file(f"{base_url}/{rec_id}.dat", os.path.join(dl_dir, f"{rec_id}.dat"))

        # Download the annotation file (.atr)
        download_file(f"{base_url}/{rec_id}.atr", os.path.join(dl_dir, f"{rec_id}.atr"))

        if i % 5 == 0:
            print(f"Downloaded {i} records...")

    print("Download complete!")


if __name__ == "__main__":
    main()
