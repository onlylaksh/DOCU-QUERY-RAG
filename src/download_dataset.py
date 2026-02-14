import kagglehub
import os

def download_data():
    """
    Downloads the dataset from Kaggle.
    Returns the absolute path to the dataset folder.
    """
    print("Initializing Kaggle dataset download...")
    path = kagglehub.dataset_download("manisha717/dataset-of-pdf-files")
    print(f"Dataset downloaded successfully to: {path}")
    return path

if __name__ == "__main__":
    download_data()
