from huggingface_hub import list_repo_files

archivos = list_repo_files("charlie-charl/treasureflow-moderation-beto")
print(archivos)