import os
import pandas as pd
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Input/output
input_dir = "data"
output_root = "./"
os.makedirs(output_root, exist_ok=True)
os.makedirs(os.path.join(output_root, "train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "test"), exist_ok=True)

# Thread-safe index counters
index_lock = Lock()
index_map = {
    "train": 0,
    "test": 0,
}

# Decode image from base64 or bytes
def decode_image(data):
    if isinstance(data, str):
        data = base64.b64decode(data)
    return Image.open(io.BytesIO(data))

# Process a single parquet file
def process_parquet(file_path):
    file_name = os.path.basename(file_path)
    mode = "train" if file_name.startswith("train-") else "test" if file_name.startswith("dev-") else None

    if mode is None:
        print(f"[!] Skipping unknown split file: {file_name}")
        return

    print(f"[+] Processing {file_path} as {mode}")
    df = pd.read_parquet(file_path)

    for _, row in df.iterrows():
        with index_lock:
            current_index = index_map[mode]
            index_map[mode] += 1

        subdir = os.path.join(output_root, mode, f"{current_index:06d}")
        os.makedirs(subdir, exist_ok=True)

        for key in ["source_img", "mask_img", "target_img"]:
            img = decode_image(row[key]['bytes'])
            img.save(os.path.join(subdir, f"{key}.png"))

        with open(os.path.join(subdir, "instructions.txt"), "w", encoding="utf-8") as f:
            f.write(row["instruction"].strip())

    print(f"[✓] Done: {file_name} ({len(df)} samples)")

# Gather all parquet files
parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")]

# Run in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_parquet, path) for path in parquet_files]
    for future in futures:
        future.result()  # Raise any exceptions

print("[✔] All files processed and saved by split.")
