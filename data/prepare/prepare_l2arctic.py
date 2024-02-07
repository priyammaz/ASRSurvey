import os
import zipfile
from tqdm import tqdm

path_to_root = "data/l2arctic"
zip_file = "l2arctic_release_v5.0.zip"
path_to_zip = os.path.join(path_to_root, zip_file)

print("Unzipping File")
with zipfile.ZipFile(path_to_zip, "r") as f:
    f.extractall(path=path_to_root) 

zip_files = [file for file in os.listdir(path_to_root) if ".zip" in file]
zip_files.remove(zip_file)

for file in tqdm(zip_files):
    with zipfile.ZipFile(os.path.join(path_to_root, file), "r") as f:
        f.extractall(path=path_to_root)

for file in zip_files:
    os.remove(os.path.join(path_to_root, file))