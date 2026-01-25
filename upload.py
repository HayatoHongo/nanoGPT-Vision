from huggingface_hub import upload_folder, HfApi
import os

repo_id = "YOURNAME/YOUREPO" # "HayatoHongo/everyoneschat-checkpoints"
local_dir = "/home/ubuntu/nanoGPT-Vision/checkpoints"

# repo create (skip if already created)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=repo_id,
    private=False,   # set True to make the repo private
    exist_ok=True
)

# Upload the entire folder
upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    token=os.environ["HF_TOKEN"],
)
