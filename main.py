import os
import tensorflow as tf
import numpy as np
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("arnaud58/selfie2anime")

print("Path to dataset files:", path)

source_path = path

current_working_dir = os.getcwd()


shutil.copytree(source_path, os.path.join(current_working_dir, 'selfie2anime'))

print("Dataset copied to current working directory.")
