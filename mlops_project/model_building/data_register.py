from huggingface_hub import HfApi, create_repo
import os

repo_id = "ashwindatasense/wellness-tourism-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except Exception:
    print(f"Space '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_folder(
    folder_path="mlops_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)
print("Data uploaded successfully!")
