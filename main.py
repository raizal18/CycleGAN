import os
import tensorflow as tf
import shutil
import kagglehub
import logging
from loader import load_dataset

tf.config.set_visible_devices([], 'GPU')  # Disable GPU


log_file = "log_file.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

logger = logging.getLogger()


try:
    logger.info("Downloading the dataset from Kaggle...")
    path = kagglehub.dataset_download("arnaud58/selfie2anime")
    logger.info(f"Path to dataset files: {path}")
except Exception as e:
    logger.error(f"Failed to download dataset: {str(e)}")
    raise


# source_path = path
# current_working_dir = os.getcwd()

# try:
#     logger.info(f"Copying dataset from {source_path} to {current_working_dir}...")
#     shutil.copytree(source_path, os.path.join(current_working_dir, 'selfie2anime'),dirs_exist_ok=True)
#     logger.info("Dataset copied to current working directory.")
# except Exception as e:
#     logger.error(f"Failed to copy dataset: {str(e)}")
#     raise

batch_size = 32
image_size = (256, 256)
dataset_path = path

try:
    logger.info(f"Loading dataset from {dataset_path}...")
    train_dataset, test_dataset = load_dataset(dataset_path, image_size, batch_size)
    logger.info(f"Dataset loaded successfully. Train dataset size: {len(list(train_dataset))}, Test dataset size: {len(list(test_dataset))}")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    raise

try:
    logger.info(f"Loading dataset from {dataset_path}...")
    train_dataset, test_dataset = load_dataset(dataset_path, image_size, batch_size)
    logger.info(f"Dataset loaded successfully. Train dataset size: {len(list(train_dataset))}, Test dataset size: {len(list(test_dataset))}")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    raise
