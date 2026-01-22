import os
import subprocess

def download_from_kaggle():
    dataset = 'hridoyyahmed/poribohon-bd'
    os.makedirs('data', exist_ok=True)
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '--unzip', '-p', 'data/poribohon-bd'])
    print("Dataset downloaded and extracted to data/poribohon-bd")


def manual_download():
    url = "https://www.kaggle.com/datasets/hridoyyahmed/poribohon-bd/download?datasetVersionNumber=1"  
    print(f"Download manually from {url} and extract to data/poribohon-bd")

if __name__ == "__main__":
    manual_download() 