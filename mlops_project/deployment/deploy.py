from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

api = HfApi(token=hf_token)
user_info = api.whoami()
username = user_info['name']

space_id = f"{username}/wellness-tourism-app"

print(f"Checking if Space '{space_id}' exists...")
try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print("Space already exists.")
except Exception:
    print("Creating new Streamlit Space...")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=hf_token
    )
    print("Space created!")

print("Uploading deployment files...")
api.upload_folder(
    folder_path="mlops_project/deployment",
    repo_id=space_id,
    repo_type="space",
    token=hf_token
)
print("Deployment files uploaded successfully! Your app is building on Hugging Face Spaces.")
