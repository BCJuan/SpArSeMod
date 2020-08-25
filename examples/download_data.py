import requests
import os
import shutil
import tqdm
from clint.textui import progress
import zipfile

ROOT="./data"
CIFAR2="data_cifar2"
COST="data_cost"
CIFAR_ID_DRIVE="0B5E8qFcWFPQOYXg3X29uMDdINU0"
COST_LINK = "https://data.4tu.nl/ndownloader/articles/12696869/versions/1"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_cost():

    if not os.path.exists(os.path.join(ROOT, COST, "CoST.csv")):
        if not os.path.exists(os.path.join(ROOT, COST, "cost.zip")):
            myfile = requests.get(COST_LINK, allow_redirects=True, stream=True)

            with open(os.path.join(ROOT, COST, "cost.zip"), 'wb') as f:
                total_length = int(myfile.headers.get('content-length'))
                for chunk in progress.bar(myfile.iter_content(chunk_size = 2391975), expected_size=(total_length/1024) + 1):
                    if chunk:
                        f.write(chunk)
        
            with zipfile.ZipFile(os.path.join(ROOT, COST, "cost.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(ROOT, COST))
            os.remove(os.path.join(ROOT, COST, "cost.zip"))

def download_cifar2():
    if not os.path.exists(os.path.join(ROOT, CIFAR2, "cifar10binary.train")):
        if not os.path.exists(os.path.join(ROOT, CIFAR2, "cifar2.zip")):
            download_file_from_google_drive(CIFAR_ID_DRIVE, os.path.join(ROOT, CIFAR2, "cifar2.zip"))
            with zipfile.ZipFile(os.path.join(ROOT, CIFAR2, "cifar2.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(ROOT, CIFAR2))
            os.remove(os.path.join(ROOT, CIFAR2, "cifar2.zip"))
        for i in os.listdir(os.path.join(ROOT, CIFAR2, "datasets")):
            if "cifar" in i:
                shutil.move(os.path.join(ROOT, CIFAR2, "datasets", i), os.path.join(ROOT, CIFAR2,i))
        shutil.rmtree(os.path.join(ROOT, CIFAR2, "datasets"))

if __name__ == "__main__":
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    cifar2_dir = os.path.join(ROOT, CIFAR2)
    if not os.path.exists(cifar2_dir):
        os.mkdir(cifar2_dir)
    cost_dir = os.path.join(ROOT, COST)
    if not os.path.exists(cost_dir):
        os.mkdir(cost_dir)
    
    download_cost()
    download_cifar2()
