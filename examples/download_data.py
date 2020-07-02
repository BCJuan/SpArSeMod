import requests
import os

ROOT="./data"
CIFAR2="data_cifar2"
COST="data_cost"


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


if __name__ == "__main__":
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    cifar2_dir = os.path.join(ROOT, CIFAR2)
    if not os.path.exists(cifar2_dir):
        os.mkdir(cifar2_dir)
    cost_dir = os.path.join(ROOT, COST)
    if not os.path.exists(cost_dir):
        os.mkdir(cost_dir)
    
    download_url("https://drive.google.com/u/0/uc?export=download&confirm=nfG5&id=0B5E8qFcWFPQOYXg3X29uMDdINU0", "cifar2.zip")