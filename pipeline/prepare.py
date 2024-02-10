import os
import tarfile 
import zipfile 
from utils import part_number_gen, download
from tqdm import tqdm
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class DataLinks:
    ### CORAAL DATASET ###
    CORAAL_ATL_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part{part}_2020.05.tar.gz" for part in part_number_gen(4)], 
                    "transcripts": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_textfiles_2020.05.tar.gz", 
                    "metadata": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_metadata_2020.05.txt"}

    CORAAL_DCA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(10)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textfiles_2018.10.06.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_metadata_2018.10.06.txt"}


    CORAAL_DCB_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(14)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_textfiles_2018.10.06.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_metadata_2018.10.06.txt"}

    CORAAL_DTA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part{part}_2023.06.tar.gz" for part in part_number_gen(10)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_textfiles_2023.06.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_metadata_2023.06.txt"}

    CORAAL_LES_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part{part}_2021.07.tar.gz" for part in part_number_gen(3)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_textfiles_2021.07.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt"}

    CORAAL_PRV_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(4)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_textfiles_2018.10.06.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_metadata_2018.10.06.txt"}

    CORAAL_ROC_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part{part}_2020.05.tar.gz" for part in part_number_gen(5)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_textfiles_2020.05.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_metadata_2020.05.txt"}

    CORAAL_VLD_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part{part}_2021.07.tar.gz" for part in part_number_gen(4)], 
                        "transcripts": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_textfiles_2021.07.tar.gz", 
                        "metadata": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt"}


    coraal = {"atlanta_georgia": CORAAL_ATL_FILES, 
            "washington_dc_1968": CORAAL_DCA_FILES,
            "washington_dc_2016": CORAAL_DCB_FILES, 
            "detroit_michicagn": CORAAL_DTA_FILES, 
            "lower_east_side_new_york": CORAAL_LES_FILES, 
            "princeville_north_carolina": CORAAL_PRV_FILES, 
            "rochester_new_york": CORAAL_ROC_FILES, 
            "valdosta_georgia": CORAAL_VLD_FILES}

    ### EDACC Dataset ###
    edacc = "https://datashare.ed.ac.uk/bitstream/handle/10283/4836/edacc_v1.0.tar.gz"



class PrepareCoraal:

    """

    Data Download for entire CORAAL Dataset:

    Link: http://lingtools.uoregon.edu/coraal/

    Citation:
    "Kendall, Tyler and Charlie Farrington. 2023. The Corpus of Regional African American Language. 
    Version 2023.06. Eugene, OR: The Online Resources for African American Language Project."

    """

    def __init__(self, path_to_root="data/coraal"):
        self.path_to_root = path_to_root
        self.coraal = DataLinks.coraal

        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(path_to_root)
    
    def prepare(self, delete_extra_files=True):
        """
        delete_extra_files:  Remove all downloaded/compressed files and only keep uncompressed data
        """
        for location in self.coraal:
            print(f"Processing {location}")
            path_to_loc_folder = os.path.join(self.path_to_root, location)
            path_to_audio_store = os.path.join(path_to_loc_folder, "audios")
            path_to_transcript_store = os.path.join(path_to_loc_folder, "transcripts")
            if not os.path.isdir(path_to_loc_folder):
                os.mkdir(path_to_loc_folder)
                os.mkdir(path_to_audio_store)
                os.mkdir(path_to_transcript_store)

            for audio in tqdm(self.coraal[location]["audio"]):
                audio_file_name = audio.split("/")[-1]
                path_to_audio_file = os.path.join(path_to_audio_store, audio_file_name)

                ### Download Audios ###
                if not os.path.exists(path_to_audio_file):
                    download(audio, path_to_audio_file)
            
                ### Decompress Audios ###
                with tarfile.open(path_to_audio_file) as f:
                    f.extractall(path_to_audio_store)

            ### Download and Decompress Transcript ###
            transcript_link = self.coraal[location]["transcripts"]
            transcript_name = transcript_link.split("/")[-1]
            path_to_trancript_file = os.path.join(path_to_transcript_store, transcript_name)

            download(transcript_link, path_to_trancript_file)
            with tarfile.open(path_to_trancript_file) as f:
                f.extractall(path_to_transcript_store)

            ### Download MetaData ###
            download(self.coraal[location]["metadata"], os.path.join(path_to_loc_folder, "metadata.txt"))


            ### Delete and Hidden and Compressed Files ###
            if delete_extra_files:
                for file in os.listdir(path_to_audio_store):
                    if file.startswith(".") or ".tar.gz" in file:
                        os.remove(os.path.join(path_to_audio_store, file))

                for file in os.listdir(path_to_transcript_store):
                    if file.startswith(".") or ".tar.gz" in file:
                        os.remove(os.path.join(path_to_transcript_store, file))

class PrepareEdacc:

    """
    Download for EDACC Dataset 

    Link: https://groups.inf.ed.ac.uk/edacc/

    Citation:
    "The Edinburgh International Accents of English Corpus: Towards the Democratization of
    English ASR. Ramon Sanabria, Bogoychev, Markl, Carmantini, Klejch, and Bell. ICASSP 2023."

    """

    def __init__(self, path_to_root="data/edacc"):
        self.path_to_root = path_to_root
        self.edacc = DataLinks.edacc

        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(path_to_root)

    def prepare(self, delete_extra_files=False):
        """
        delete_extra_files:  Remove all downloaded/compressed files and only keep uncompressed data
        """
        print("FYI: EDACC Dataset can take some time to download")
        print("Downloading Data")
        path_to_download = os.path.join(self.path_to_root, "edacc.tar.gz")
        download(self.edacc, path_to_download, progress_bar=True)

        print("Decompressing Data")
        with tarfile.open(path_to_download) as f:
            f.extractall(self.path_to_root)

        if delete_extra_files:
            os.remove(path_to_download)

class PrepareL2Arctic:

    """
    
    Preparation of the L2Arctic Dataset, cant download from here due to access requirement, access can be 
    done here:

    Link: https://psi.engr.tamu.edu/l2-arctic-corpus/

    Citation: 
    Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J., Gutierrez-Osuna, R. (2018) 
    L2-ARCTIC: A Non-native English Speech Corpus. Proc. Interspeech 2018, 2783-2787, doi: 10.21437/Interspeech.2018-1110

    """
    def __init__(self, 
                 path_to_root="data/l2arctic", 
                 downloaded_zipfile_name="l2arctic_release_v5.0.zip"):
        
        self.path_to_root = path_to_root
        self.zipfile = downloaded_zipfile_name
        
        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            print("Drop downloaded L2Arctic Zip file into the directory")
            os.mkdir(path_to_root)

    def prepare(self, delete_extra_files=True):
        """
        delete_extra_files:  Remove all downloaded/compressed files and only keep uncompressed data
        """
        path_to_zip = os.path.join(self.path_to_root, self.zipfile)
        assert os.path.isfile(path_to_zip), "Get access and download data from \
            https://psi.engr.tamu.edu/l2-arctic-corpus/ and place in root directory"

        
        print("Unzipping File")
        with zipfile.ZipFile(path_to_zip, "r") as f:
            f.extractall(path=self.path_to_root) 

        zip_files = [file for file in os.listdir(self.path_to_root) if ".zip" in file]
        zip_files.remove(self.zipfile)

        for file in tqdm(zip_files):
            with zipfile.ZipFile(os.path.join(self.path_to_root, file), "r") as f:
                f.extractall(path=self.path_to_root)

        if delete_extra_files:
            for file in zip_files:
                os.remove(os.path.join(self.path_to_root, file))

        
class PrepareSpeechAccentArchive:

    """

    Preparation of the Speech Accent Archive, cant download due to log-in requirements, manually download 
    and place zip from Kaggle into root directory

    Link: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

    Citation: 
    https://accent.gmu.edu/about.php

    """

    def __init__(self, 
                 path_to_root="data/speech_accent_archive", 
                 downloaded_zipfile_name="archive.zip"):
        
        self.path_to_root = path_to_root
        self.zipfile = downloaded_zipfile_name
        
        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            print("Drop downloaded Speech Accent Archive Zip file into the directory")
            os.mkdir(path_to_root)

    def prepare(self, delete_extra_files=True):
        """
        delete_extra_files:  Remove all downloaded/compressed files and only keep uncompressed data
        """

        path_to_zip = os.path.join(self.path_to_root, self.zipfile)
        assert os.path.isfile(path_to_zip), "Get access and download data from \
            https://www.kaggle.com/datasets/rtatman/speech-accent-archive and place in root directory"
        
        print("Unzipping File")
        with zipfile.ZipFile(path_to_zip, "r") as f:
            f.extractall(path=self.path_to_root) 

        if delete_extra_files:
            os.remove(path_to_zip)

class PrepareMozillaCommonVoice:

    """
    Prepare entire Mozilla CommonVoice 13.0 Dataset from Huggingface

    Link: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0

    Citation: 
    https://commonvoice.mozilla.org/en/datasets

    """

    def __init__(self, 
                 path_to_root="data/mozilla"):
        
        self.path_to_root = path_to_root
        
        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(path_to_root)

    def prepare(self, num_workers=8):
        print("This will take some time and about 130GB Hard Drive Space!")
        dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", 
                               cache_dir=self.path_to_root, 
                               num_proc=num_workers)


        print("Preprocessing to Remove Samples with Unlabeled Accents. THIS WILL TAKE SOME TIME!!!")

        dataset = dataset.rename_column("sentence", "transcription")
        dataset = dataset.remove_columns(["client_id", "path", "up_votes", "down_votes",
                                        "locale", "segment", "variant"])
        dataset = dataset.filter(lambda example: example["accent"]!="", num_proc=num_workers)
        
        print("Save the Filtered Dataset to Disk")
        dataset.save_to_disk(os.path.join(self.path_to_root, "accented_mozilla.hf"))



if __name__ == "__main__":
    PrepareMozillaCommonVoice().prepare()