import os
import numpy as np
import re
import tarfile 
import zipfile 
from utils import part_number_gen, download
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import librosa
import soundfile as sf
from datasets import load_dataset, Audio
from config import DatasetConfig as dc
from config import InferenceConfig as ic

###################################
####### DATASET PREPARATION #######
###################################


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

    def __init__(self, config=dc.dataset_catalog["CORAAL"]):
        self.config = config
        self.path_to_root = config["path_to_data"]
        self.coraal = DataLinks.coraal

        self.sr = 16000
        self.location_dirs = ['atlanta_georgia', 'detroit_michicagn', 'lower_east_side_new_york',  \
                              'princeville_north_carolina', 'rochester_new_york', 'valdosta_georgia', \
                              'washington_dc_1968', 'washington_dc_2016']

        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(self.path_to_root)
    
    def _gen_dataset(self, location):
        audio_root = os.path.join(self.path_to_root, location, "audios")
        transcription_root = os.path.join(self.path_to_root, location, "transcripts")
        path_to_metadata = os.path.join(self.path_to_root, location, "metadata.txt")

        file_roots = [file for file in os.listdir(audio_root) if ".wav" in file]
        file_roots = [file.split(".")[0] for file in file_roots if not file.startswith(".")]
        path_to_files = [(os.path.join(audio_root, f"{file}.wav"), os.path.join(transcription_root, f"{file}.txt")) for file in file_roots]

        cleaned_transcription = []
        for path_to_audio, path_to_transcriptions in path_to_files:
            filetag = path_to_audio.split("/")[-1].split(".")[0]
            transcriptions = pd.read_csv(path_to_transcriptions, sep='\t', lineterminator='\n')
            transcriptions = transcriptions[transcriptions["Spkr"].str.contains("int")==False]
            transcriptions = transcriptions[transcriptions["Content"].str.contains("pause")==False]
            transcriptions = transcriptions[transcriptions["Content"].str.contains("<|>|RD-")==False]
            transcriptions["Content"] = transcriptions["Content"].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()
            transcriptions["total_time"] = transcriptions["EnTime"] - transcriptions["StTime"]
            transcriptions = transcriptions[transcriptions["total_time"] >= 2].reset_index(drop=True)
            transcriptions["start_frame"] = (transcriptions["StTime"]*self.sr).astype(int)
            transcriptions["end_frame"] = (transcriptions["EnTime"]*self.sr).astype(int)
            transcriptions["path_to_audio"] = path_to_audio
            transcriptions = transcriptions[["start_frame", "end_frame", "Content", "path_to_audio"]]
            transcriptions["file_id"] = filetag
            cleaned_transcription.append(transcriptions)
        
        transcriptions = pd.concat(cleaned_transcription).reset_index(drop=True) 
        metadata = pd.read_csv(path_to_metadata,sep='\t', lineterminator='\n')
        data = pd.merge(transcriptions, metadata, left_on="file_id", right_on="CORAAL.File", how="left")
        selected_columns = ["start_frame", "end_frame", "Content", "path_to_audio", 
                            "file_id", "Gender", "Age", "Occupation"]
        
        data = data[selected_columns]
        data.columns = ["start_frame", "end_frame", "transcription", "path_to_audio", 
                        "file_id", "gender", "age", "occupation"]
        
        data["location"] = " ".join(location.split("_"))
        data["accent"] = "african_american"

        return data

        

    def prepare(self, cut_audios=True, delete_extra_files=True):
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

        if cut_audios:

            ### Build CSV with All Data and Start/End Times in Audios ###
            datasets = []
            for location in self.location_dirs:
                dataset = self._gen_dataset(location)
                datasets.append(dataset)

            datasets = pd.concat(datasets).reset_index(drop=True)

            ### Create Unique Identifier for Each Audio File ###
            datasets["audio_id"] = ""
            for grouper, group in datasets.groupby("file_id"):
                datasets.loc[datasets["file_id"] == grouper, "audio_id"] = group["file_id"] + [f"_{i+1}" for i in range(len(group))]
            

            ### Iterate Through CSV and Slice up Audios ###
            print("Cutting Up Audios!!")

            datasets["path_to_audio_split"] = ""
            for grouper, group in tqdm(datasets.groupby("path_to_audio")):
                path_to_split_audio = os.path.join("/".join(grouper.split("/")[:3]), "split_audio")
                audio = librosa.load(grouper, sr=self.sr)[0]
                
                if not os.path.isdir(path_to_split_audio):
                    os.mkdir(path_to_split_audio)

                for idx, row in group.iterrows():
                    audio_start, audio_end = row.start_frame, row.end_frame
                    path_to_audio_store = os.path.join(path_to_split_audio, f"{row.audio_id}.wav")
                    datasets.loc[idx, "path_to_audio_split"] = path_to_audio_store

                    audio_split = audio[audio_start:audio_end]

                    sf.write(path_to_audio_store, audio_split, self.sr)

            ### Save Datasets as Metadata ###
            datasets.to_csv(os.path.join(self.path_to_root, "metadata.csv"), index=False)

class PrepareEdacc:

    """
    Download for EDACC Dataset 

    Link: https://groups.inf.ed.ac.uk/edacc/

    Citation:
    "The Edinburgh International Accents of English Corpus: Towards the Democratization of
    English ASR. Ramon Sanabria, Bogoychev, Markl, Carmantini, Klejch, and Bell. ICASSP 2023."

    """

    def __init__(self, config=dc.dataset_catalog["EDACC"]):
        self.config = config
        self.path_to_root = config["path_to_data"]
        self.edacc = DataLinks.edacc

        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(self.path_to_root)

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
                 config=dc.dataset_catalog["L2Arctic"]):
        
        self.config = config
        self.path_to_root = config["path_to_data"]
        self.zipfile = config["download_file_name"]
        
        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            print("Drop downloaded L2Arctic Zip file into the directory")
            os.mkdir(self.path_to_root)

        assert os.path.isfile(os.path.join(self.path_to_root, self.zipfile)), f"Make sure to place {self.zipfile} in {self.path_to_root}, \
            or check path_to_data and download_file_name in DatasetConfig"

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
                 config=dc.dataset_catalog["SpeechAccentArchive"]):
        
        self.config = config
        self.path_to_root = config["path_to_data"]
        self.zipfile = config["download_file_name"]
        self.path_to_audio = os.path.join(self.path_to_root, "recordings/recordings/")
        self.sampling_rate = ic.sample_rate
        
        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            print("Drop downloaded Speech Accent Archive Zip file into the directory")
            os.mkdir(self.path_to_root)

        assert os.path.isfile(os.path.join(self.path_to_root, self.zipfile)), f"Make sure to place {self.zipfile} in {self.path_to_root}, \
            or check path_to_data and download_file_name in DatasetConfig"

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

        metadata = pd.read_csv(os.path.join(self.path_to_root, "speakers_all.csv"),
                               usecols=["age", "birthplace", "native_language", "sex", "filename"])
        metadata["path_to_audio"] = self.path_to_audio + metadata["filename"] + ".mp3"
        
        ### Check If Any Missing Files ###
        metadata["audio_check"] = metadata["path_to_audio"].map(lambda x: os.path.isfile(x))
        metadata = metadata.loc[metadata["audio_check"] == True].reset_index(drop=True)
        metadata = metadata.drop(columns="audio_check")
        
        ### Add Audio Length Column for Future Filtering ###
        print("Computing Audio Duration for All Available Files")
        audios_durations = []
        for path in tqdm(metadata["path_to_audio"]):
            y, sr = librosa.load(path)
            audios_durations.append(len(y) / sr)
            
        metadata["audio_duration"] = audios_durations
        metadata.to_csv(os.path.join(self.path_to_root, "speakers_all.csv"), index=False)


class PrepareMozillaCommonVoice:

    """
    Prepare entire Mozilla CommonVoice 13.0 Dataset from Huggingface

    Link: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0

    Citation: 
    https://commonvoice.mozilla.org/en/datasets

    """

    def __init__(self, 
                 config=dc()):
        
        self.path_to_root = config.dataset_catalog["MozillaCommonVoice"]["path_to_data"]
        self.subset_dataset_name = config.dataset_catalog["MozillaCommonVoice"]["accent_subset"]
        self.max_audio_length = config.max_audio_length
        self.sr = 16000

        ### Check Root Exists ###
        if not os.path.isdir(self.path_to_root):
            print("Creating Root Directory")
            os.mkdir(self.path_to_root)

    def compute_length(self, example):
        example["audio_length"] = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
        return example

    def prepare(self, num_workers=8):
        print("This will take some time and about 130GB Hard Drive Space!")
        dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", 
                               cache_dir=self.path_to_root, 
                               num_proc=num_workers)

        print("Preprocessing to Remove Samples with Unlabeled Accents and Long Audio. THIS WILL TAKE SOME TIME!!!")

        ### Filter Out Long Audio 
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sr))
        dataset = dataset.filter(lambda example: (len(example["audio"]["array"]) / example["audio"]["sampling_rate"]) < 30, num_proc=num_workers)
        
        ### Filter Out Unaccented Data or Non-Upvoted/Downvoted Data ###
        dataset = dataset.filter(lambda example: (example["accent"]!="") or (example["up_votes"]<4) or (example["down_votes"]>0), num_proc=num_workers)
        
        ### Remove Extra Columns ###
        dataset = dataset.rename_column("sentence", "transcription")
        dataset = dataset.remove_columns(["client_id", "path", "up_votes", "down_votes",
                                        "locale", "segment", "variant"])
        
        print("Save the Filtered Dataset to Disk")
        dataset.save_to_disk(os.path.join(self.path_to_root, "accented_mozilla.hf"))


class PrepareSantaBarbara:
    def __init__(self, config=dc()):
        self.path_to_root = config.dataset_catalog["SantaBarbara"]["path_to_data"]
        self.path_to_store = os.path.join(self.path_to_root, "sliced_audio")

        if not os.path.isdir(self.path_to_store):
            os.mkdir(self.path_to_store)

        self.max_audio_length = config.max_audio_length
        self.sr = 16000

    def prepare(self, min_words=3):

        remove_parenthesis = re.compile(r'\([^()]*\)')
        remove_noncharacters = re.compile('[^a-zA-Z ]+')
        remove_dashed_characters = re.compile("[A-Z]-")


        data_dict = {"path_to_audio": [], 
                     "transcript": [],
                     "reader": [],
                     "duration": []}
        

        for f in ["sbcsae", "sbcsae_2", "sbcsae_4", "sbcsae_p3"]:
            print(f)
            print("Parsing", f)
            path_to_sub = os.path.join(self.path_to_root, f, "speech")
            audio_files = [w for w in os.listdir(path_to_sub) if ".wav" in w ]
            audio_transcript_pairs = [(os.path.join(path_to_sub, a),os.path.join(path_to_sub, a.split(".")[0]+".trn")) for a in audio_files]
            valid_audios = []
            for pair in audio_transcript_pairs:
                if os.path.isfile(pair[0]) and os.path.isfile(pair[1]):
                    valid_audios.append(pair)

            for audio, transcript in tqdm(valid_audios):
                try:
                    transcript = pd.read_csv(transcript, sep='\t', lineterminator='\n', header=None, on_bad_lines='skip')
                    if f == "sbcsae":
                        transcript.columns = ["time", "reader", "transcript"]
                        transcript[["start", "end"]] = transcript["time"].str.split(" ", expand=True)
                        transcript = transcript.drop(columns="time")

                    else:
                        transcript.columns = ["start", "end", "reader", "transcript"]
                    
                    ### Fill out Readers ###
                    transcript["reader"] = transcript["reader"].str.strip().replace("", np.nan)
                    transcript["reader"] = transcript["reader"].ffill()
                    ### Remove End Characters and tags between () ###
                    transcript["transcript"] = transcript["transcript"].str.replace("\r", "")
                    transcript["transcript"] = [remove_parenthesis.sub("",str(i)) for i in transcript["transcript"]]

                    ### Filter out Sentences with blanked words ###
                    transcript["transcript"] = [i if "~" not in i else "" for i in transcript["transcript"]]

                    ### Remove Dashed Characters ###
                    transcript["transcript"] = [remove_dashed_characters.sub("",str(i)) for i in transcript["transcript"]]

                    ### Remove all non characters ###
                    transcript["transcript"] = [remove_noncharacters.sub("",str(i)).lower() for i in transcript["transcript"]]
                    print(transcript)
                    ### Remove all 1 Letter words (except a and i) and final cleanup ###
                    cleaned_transcript = []
                    for t in transcript["transcript"]:
                        t = t.strip().split()
                        new_t = []
                        for w in t:
                            if len(w) == 1:
                                if w in ["a", "i"]:
                                    new_t.append(w)
                            else:
                                new_t.append(w)
                                

                        t = " ".join(new_t)
                        cleaned_transcript.append(t)
                        
                    transcript["transcript"] = cleaned_transcript

                    transcript["word_count"] = transcript["transcript"].apply(lambda x: len(x.split()))

                    transcript = transcript.loc[transcript["word_count"] >= min_words]

                    ### Load Audio ###
                    y, sr = librosa.load(audio, sr=self.sr)
                    
                    for idx, row in transcript.iterrows():
                        start_frame = int(float(row.start) * sr)
                        end_frame = int(float(row.end) * sr)
                        
                        audio_split = y[start_frame:end_frame]

                        save_path = os.path.join(self.path_to_store, f"{audio.split("/")[-1].split(".")[0]}_{idx}.wav")
                        
                        data_dict["path_to_audio"].append(save_path)
                        data_dict["transcript"].append(row.transcript)
                        data_dict["reader"].append(row.reader)
                        data_dict["duration"].append((float(row.end) - float(row.start)))

                        sf.write(save_path, audio_split, sr)

                except Exception as e:
                    print(e)
                    print("FAILED", audio)
    

        metadata = pd.DataFrame(data_dict)
        metadata.to_csv(os.path.join(self.path_to_root, "metadata.csv"), index=False)
                    
            


###################################
######## MODEL PREPARATION ########
###################################

# class DownloadModels:
#     def __init__(self, 
#                  model_configs=ic,
#                  only_inference = None, 
#                  exclude_inference = None,
#                  limit_parameter_size = None):
        
        

if __name__ == "__main__":
    # PrepareMozillaCommonVoice().prepare()
    PrepareSantaBarbara().prepare()
