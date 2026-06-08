from huggingface_hub import login, upload_folder

login()
upload_folder(
    folder_path="model",
    repo_id="toanle1355/skin_binary",
    repo_type="model",
)

import os
access_token = os.environ.get("HF_TOKEN")  # Use environment variable
# Push your model files
