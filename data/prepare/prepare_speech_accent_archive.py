import os
import zipfile
from tqdm import tqdm

path_to_root = "data/speech_accent_archive/"
path_to_zip = os.path.join(path_to_root, "archive.zip")

### Unzip File ###
print("Unzipping File")
with zipfile.ZipFile(path_to_zip, "r") as f:
    f.extractall(path=path_to_root) 

os.remove(path_to_zip)