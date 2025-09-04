import kagglehub

# Download latest version
path = kagglehub.dataset_download("swekerr/click-through-rate-prediction")

print("Path to dataset files:", path)