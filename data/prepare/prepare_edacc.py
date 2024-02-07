import requests 
import os
import tarfile
from zipfile import ZipFile 
from tqdm import tqdm 

dataset = {"metadata": "https://datashare.ed.ac.uk/bitstream/handle/10283/4836/linguistic_background.csv?sequence=1&isAllowed=y", 
           "files": "https://datashare.ed.ac.uk/bitstream/handle/10283/4836/edacc_v1.0.tar.gz?sequence=4&isAllowed=y"}

path_to_root = "data/edacc"
if not os.path.isdir(path_to_root):
    os.mkdir(path_to_root)


### Download Dataset ###
print("Downloading Data")
path_to_download = os.path.join(path_to_root, "edacc_v1.0.tar.gz")
response = requests.get(dataset["files"], stream=True)
total = int(response.headers.get('content-length', 0))

with open(path_to_download, 'wb') as file, tqdm(
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
### Decompress Data ###
print("Decompressing Data")
with tarfile.open(path_to_download) as f:
    f.extractall(path_to_root)