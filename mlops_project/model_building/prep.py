import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "ashwindatasense/wellness-tourism-dataset"

# Construct direct download URL for the raw dataset
base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"

# Load dataset using HTTPS instead of hf://
print("Loading raw dataset from Hugging Face Hub...")
df = pd.read_csv(f"{base_url}/tourism.csv")

# Clean
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Split
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
X_train.to_csv(f"{data_dir}/X_train.csv", index=False)
X_test.to_csv(f"{data_dir}/X_test.csv", index=False)
y_train.to_csv(f"{data_dir}/y_train.csv", index=False)
y_test.to_csv(f"{data_dir}/y_test.csv", index=False)

# Upload
for file_name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=f"{data_dir}/{file_name}",
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset"
    )
print("Data prep complete and split files uploaded.")
