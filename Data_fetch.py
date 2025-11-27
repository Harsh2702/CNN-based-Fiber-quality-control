import os
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------- CONFIG ----------------
CONNECTION_STRING = "<Connection String>"
CONTAINER_NAME = "Conatainername"
FOLDER_NAMES = [
    "F1", "F2", "F3", "F10"
]

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


# ---------------- STEP 1: FIND EXCEL FILES ----------------
def list_auswertung_files():
    ps2_files, ps3_files = [], []
    for folder_path in FOLDER_NAMES:
        for blob in container_client.list_blobs(name_starts_with=folder_path):
            # Check if 'Auswertung' is in the blob path, and ends with .xlsx
            if "auswertung" in blob.name.lower() and blob.name.lower().endswith(".xlsx"):
                if "ps2" in blob.name.lower():
                    ps2_files.append(blob.name)
                elif "ps3" in blob.name.lower():
                    ps3_files.append(blob.name)
    print(f"PS2: {len(ps2_files)} files, PS3: {len(ps3_files)} files")
    return ps2_files, ps3_files


# ---------------- STEP 2: PARSE LABELS ----------------
def parse_excel_labels(file_path):
    """
    Parse Excel and return IO (label=0) and NIO (label=1 pseudo / label=2 NIO) DataFrames.
    NIO section: first 3 columns marked=1 → pseudo, rest columns marked=1 → NIO
    """
    import pandas as pd
    import numpy as np

    # --- Read Excel from Azure ---
    blob_client = container_client.get_blob_client(file_path)
    data = blob_client.download_blob().readall()
    df = pd.read_excel(BytesIO(data))

    # --- Skip metadata rows ---
    df = df.iloc[3:].reset_index(drop=True)
    io_df = df.iloc[:, :2].reset_index(drop=True)
    nio_df = df.iloc[:, 2:].reset_index(drop=True)

    # --- Helper function for section processing ---
    def process_section_binary(section_df, suffix, label_value):
        section_df.columns = section_df.iloc[0]
        section_df = section_df[1:].reset_index(drop=True)
        section_df.columns = (
            section_df.columns.str.replace("\n", "", regex=False).str.strip()
        )
        section_df = section_df.dropna(subset=["Datei-name"]).reset_index(drop=True)

        base_dir = os.path.dirname(file_path)
        base_prefix = f"{base_dir}/{suffix}"
        records = []

        folder = file_path.lower()

        for _, row in section_df.iterrows():
            fname = str(row["Datei-name"]).strip()
            blob_path = f"{base_prefix}/{fname}"

            # --- IO section ---
            if label_value == 0:
                records.append({"blob_path": blob_path, "label": 0})
                continue

            # --- NIO section: convert 'x' to 1, NaN to 0 ---
            df_processed = row[2:].apply(lambda x: 1 if str(x).lower() == 'x' else 0)

            # Determine first 3 vs remaining columns
            first3_cols = df_processed.iloc[:3]
            rest_cols = df_processed.iloc[3:]

            # Assign label based on presence of 1
            if first3_cols.sum() > 0:
                row_label = 1  # Pseudo
            elif rest_cols.sum() > 0:
                row_label = 2  # NIO
            else:
                row_label = 0  # no defect marked

            records.append({"blob_path": blob_path, "label": row_label})

        return pd.DataFrame(records)

    # --- Process IO and NIO ---
    io_out = process_section_binary(io_df, "IO", 0)
    nio_out = process_section_binary(nio_df, "NIO", 1)

    # --- Summary ---
    # combined = pd.concat([io_out, nio_out], ignore_index=True)
    # counts = combined["label"].value_counts().sort_index()
    # print(f"\n✅ Parsed {file_path}")
    # print(f"   IO rows: {len(io_out)}, NIO rows: {len(nio_out)}")
    # print(f"   Label counts: {counts.to_dict()}")

    return io_out, nio_out


# ---------------- STEP 3: IMAGE READER ----------------
def read_and_preprocess(image_bytes):
    """Decode and preprocess a raw image."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, (2, 0, 1))
    return image.astype(np.float32) / 255.0


# ---------------- STEP 4: AZURE DATASET ----------------
class AzureBlobDataset(Dataset):
    """PyTorch dataset that lazily fetches images from Azure Blob."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        self.container_client = container_client

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        entry = self.records[idx]
        blob_path = entry["blob_path"]
        label = torch.tensor(entry["label"], dtype=torch.long)

        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            data = blob_client.download_blob().readall()
            img = read_and_preprocess(data)
            if img is None:
                raise ValueError
        except Exception:
            img = np.zeros((3, 512, 512), dtype=np.float32)  # fallback blank image

        img_tensor = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


# ---------------- STEP 5: AUGMENTED DATASET ----------------
class AugmentedDataset(Dataset):
    """Return original + augmented version of each sample."""
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return 2 * len(self.base_dataset)

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_dataset)
        img, label = self.base_dataset[base_idx]

        if idx >= len(self.base_dataset) and self.transform:
            img = self.transform(img)

        return img, label


# ---------------- STEP 6: AUGMENTATION ----------------
def get_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(0.1),
    ])

import sys

# ---------------- STEP 7: ENTRYPOINT ----------------
def get_azure_dataset_dynamic():
    """Build and return the full Azure dataset for PyTorch."""
    ps2_files, ps3_files = list_auswertung_files()
    iosum = 0
    niosum = 0
    # print(len(ps2_files),len(ps3_files))
    all_files = ps2_files + ps3_files
    print(f"Found {len(all_files)} total Excel files.")

    all_records = []
    for f in all_files:
        try:
            io_df, nio_df = parse_excel_labels(f)
            print(len(io_df),len(nio_df),f)
            iosum = iosum + len(io_df)
            niosum = niosum + len(nio_df)
            all_records.extend(io_df.to_dict(orient="records"))
            all_records.extend(nio_df.to_dict(orient="records"))
        except Exception as e:
            print(f"⚠️ Skipped {f}: {e}")

    print("IO total ",iosum," NIO total",niosum)
    print(f"✅ Total labeled records: {len(all_records)}")


    base_dataset = AzureBlobDataset(records=all_records)
    aug_transform = get_augmentation()
    full_dataset = AugmentedDataset(base_dataset, transform=aug_transform)

    return full_dataset
