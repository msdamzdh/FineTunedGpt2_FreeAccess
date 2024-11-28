from huggingface_hub import snapshot_download
from pathlib import Path

models = ["gpt2","xlnet-base-cased"]

for model in models:
    path = Path(f"./{model}")
    if path.exists()==False:
        snapshot_download(repo_id=model,local_dir=path)



