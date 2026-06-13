import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi
import os
import joblib

repo_id = "ashwindatasense/wellness-tourism-dataset"
model_repo_id = "ashwindatasense/wellness-tourism-model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create Model Repo if not exists
try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
except Exception:
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=False)

# Construct direct download URLs from Hugging Face
base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"

# Load Prepared Data using HTTPS instead of hf://
print("Loading prepared data from Hugging Face Hub...")
X_train = pd.read_csv(f"{base_url}/X_train.csv")
X_test = pd.read_csv(f"{base_url}/X_test.csv")
y_train = pd.read_csv(f"{base_url}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{base_url}/y_test.csv").squeeze()

numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                    'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
passthrough_features = ['CityTier', 'Passport', 'OwnCar']

numeric_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
passthrough_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
    (passthrough_transformer, passthrough_features)
)

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1)
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("Training production model...")
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(f"Prod Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save and upload model
joblib.dump(model_pipeline, "model.joblib")
api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=model_repo_id,
    repo_type="model"
)
print("Model uploaded successfully!")
