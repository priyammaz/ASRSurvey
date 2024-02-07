import requests
import tarfile
import os
from tqdm import tqdm

### Define all Links for Downloads ###
def PartNumberGen(num_parts):
    parts_idx = list(range(1, num_parts+1))
    parts = []
    for i in parts_idx:
        i = str(i)
        if len(i) == 1:
            parts.append(f"0{i}")
        else:
            parts.append(i)
    return parts

        
CORAAL_ATL_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part{part}_2020.05.tar.gz" for part in PartNumberGen(4)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_textfiles_2020.05.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_metadata_2020.05.txt"}

CORAAL_DCA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part{part}_2018.10.06.tar.gz" for part in PartNumberGen(10)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textfiles_2018.10.06.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_metadata_2018.10.06.txt"}


CORAAL_DCB_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part{part}_2018.10.06.tar.gz" for part in PartNumberGen(14)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_textfiles_2018.10.06.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_metadata_2018.10.06.txt"}

CORAAL_DTA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part{part}_2023.06.tar.gz" for part in PartNumberGen(10)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_textfiles_2023.06.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_metadata_2023.06.txt"}

CORAAL_LES_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part{part}_2021.07.tar.gz" for part in PartNumberGen(3)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_textfiles_2021.07.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt"}

CORAAL_PRV_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part{part}_2018.10.06.tar.gz" for part in PartNumberGen(4)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_textfiles_2018.10.06.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_metadata_2018.10.06.txt"}

CORAAL_ROC_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part{part}_2020.05.tar.gz" for part in PartNumberGen(5)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_textfiles_2020.05.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_metadata_2020.05.txt"}

CORAAL_VLD_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part{part}_2021.07.tar.gz" for part in PartNumberGen(4)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_textfiles_2021.07.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt"}


CORAAL = {"atlanta_georgia": CORAAL_ATL_FILES, 
          "washington_dc_1968": CORAAL_DCA_FILES,
          "washington_dc_2016": CORAAL_DCB_FILES, 
          "detroit_michicagn": CORAAL_DTA_FILES, 
          "lower_east_side_new_york": CORAAL_LES_FILES, 
          "princeville_north_carolina": CORAAL_PRV_FILES, 
          "rochester_new_york": CORAAL_ROC_FILES, 
          "valdosta_georgia": CORAAL_VLD_FILES}

### Download and Prepare Files ###
path_to_store = "data/coraal"
for location in CORAAL:
    print(f"Processing {location}")
    path_to_loc_folder = os.path.join(path_to_store, location)
    path_to_audio_store = os.path.join(path_to_loc_folder, "audios")
    path_to_transcript_store = os.path.join(path_to_loc_folder, "transcripts")
    if not os.path.isdir(path_to_loc_folder):
        os.mkdir(path_to_loc_folder)
        os.mkdir(path_to_audio_store)
        os.mkdir(path_to_transcript_store)

    for audio in tqdm(CORAAL[location]["audio"]):
        audio_file_name = audio.split("/")[-1]
        path_to_audio_file = os.path.join(path_to_audio_store, audio_file_name)

        ### Download Audios ###
        if not os.path.exists(path_to_audio_file):
            response = requests.get(audio, stream=True)
            if response.status_code == 200:
                with open(path_to_audio_file, "wb") as f:
                    f.write(response.raw.read())
            else:
                print(f"Failed to download {audio_file_name}")

        ### Decompress Audios ###
        with tarfile.open(path_to_audio_file) as f:
            f.extractall(path_to_audio_store)

    ### Download and Decompress Transcript ###
    transcript_link = CORAAL[location]["transcripts"]
    transcript_name = transcript_link.split("/")[-1]
    path_to_trancript_file = os.path.join(path_to_transcript_store, transcript_name)

    response = requests.get(transcript_link, stream=True)
    if response.status_code == 200:
        with open(path_to_trancript_file, "wb") as f:
            f.write(response.raw.read())
    else:
        print(f"Failed to download {transcript_name}")

    with tarfile.open(path_to_trancript_file) as f:
        f.extractall(path_to_transcript_store)

    ### Download MetaData ###
    response = requests.get(CORAAL[location]["metadata"], stream=True)
    path_to_metadata = os.path.join(path_to_loc_folder, "metadata.txt")
    if response.status_code == 200:
        with open(path_to_metadata, "wb") as f:
            f.write(response.raw.read())
    else:
        print(f"Failed to download {transcript_name}")

    ### Delete and Hidden and Compressed Files ###
    for file in os.listdir(path_to_audio_store):
        if file.startswith(".") or ".tar.gz" in file:
            os.remove(os.path.join(path_to_audio_store, file))

    for file in os.listdir(path_to_transcript_store):
        if file.startswith(".") or ".tar.gz" in file:
            os.remove(os.path.join(path_to_transcript_store, file))
    