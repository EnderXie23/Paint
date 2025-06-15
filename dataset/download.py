from huggingface_hub import list_repo_files, hf_hub_download
import os

def download_folder(repo_id, folder_path, local_dir, repo_type="dataset"):
    files = list_repo_files(repo_id, repo_type=repo_type)
    target_files = [f for f in files if f.startswith(folder_path)]

    os.makedirs(local_dir, exist_ok=True)

    for file in target_files:
        local_file_path = os.path.join(local_dir, os.path.relpath(file, folder_path))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        print(f"Downloading {file} -> {local_file_path}")
        downloaded = hf_hub_download(repo_id=repo_id, filename=file, repo_type=repo_type)
        os.rename(downloaded, local_file_path)

# Usage
download_folder(
    repo_id="osunlp/MagicBrush",
    folder_path="",
    local_dir="./MagicBrush",
    repo_type="dataset"
)
