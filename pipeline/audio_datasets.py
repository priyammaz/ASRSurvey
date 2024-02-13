import os
import numpy as np
import pandas as pd
import re
from config import DatasetConfig, InferenceConfig
from datasets import load_from_disk, concatenate_datasets, Audio
from dataclasses import dataclass

class Coraal:
    def __init__(self, 
                 dataset_config=DatasetConfig,
                 inference_config=InferenceConfig):

        self.name = "CORAAL"
        self.config = dataset_config.dataset_catalog[self.name]
        self.path_to_coraal = self.config["path_to_data"]

        self.location_dirs = ['atlanta_georgia', 'detroit_michicagn', 'lower_east_side_new_york',  \
                              'princeville_north_carolina', 'rochester_new_york', 'valdosta_georgia', \
                              'washington_dc_1968', 'washington_dc_2016']
        
        self.sr = inference_config.sample_rate

    def _gen_dataset(self, location):
        audio_root = os.path.join(self.path_to_coraal, location, "audios")
        transcription_root = os.path.join(self.path_to_coraal, location, "transcripts")
        path_to_metadata = os.path.join(self.path_to_coraal, location, "metadata.txt")

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

    def build_dataset(self):
        dataset = []
        for loc in self.location_dirs:
            dataset.append(self._gen_dataset(loc))

        data = pd.concat(dataset).reset_index(drop=True)
        return data


class SpeechAccentArchive:
    def __init__(self, dataset_config=DatasetConfig, **kwargs):
        
        self.name = "SpeechAccentArchive"
        self.config = dataset_config.dataset_catalog[self.name]
        self.path_to_root = self.config["path_to_data"]
        self.path_to_audio = os.path.join(self.path_to_root, "recordings/recordings/")
        self.path_to_transcript = os.path.join(self.path_to_root, "reading-passage.txt")
        self.metadata = pd.read_csv(os.path.join(self.path_to_root, "speakers_all.csv"))
        
        ### Load in Transcription ###
        self.load_transcript()

    def load_transcript(self):
        with open(self.path_to_transcript, "r") as f:
            transcript = f.readlines()[0]
        transcript = re.sub(r"[^a-zA-Z\s]", "", transcript).lower().strip()

        ### Cleanup double spaces ###
        self.transcript = " ".join(transcript.split())
        
    def build_dataset(self, max_audio_length=30):
        self.metadata["transcription"] = self.transcript
        self.metadata = self.metadata[self.metadata["audio_duration"] < max_audio_length].drop(columns="audio_duration")
        return self.metadata.reset_index(drop=True)


class Edacc:
    def __init__(self, 
                 dataset_config=DatasetConfig,
                 inference_config=InferenceConfig):
        
        self.name = "EDACC"
        self.config = dataset_config.dataset_catalog[self.name]
        self.path_to_root = self.config["path_to_data"]
        self.path_to_files = os.path.join(self.path_to_root, "edacc_v1.0")
        self.path_to_audios = os.path.join(self.path_to_files, "data")
        self.path_to_metadata = os.path.join(self.path_to_files, "linguistic_background.csv")

        self.sr = inference_config["sample_rate"]

    def build_dataset(self):
        dataset = []
        for split in ["dev", "test"]:
            path_to_split = os.path.join(self.path_to_files, split)
            transcripts = os.path.join(path_to_split, "text")
            segments = os.path.join(path_to_split, "segments")

            ### Load and Prep Transcripts ###
            with open(transcripts, "r") as f:
                transcripts = f.readlines()

            cleaned_transcript = {"segment_id": [], 
                                  "transcription": []}
            for t in transcripts:
                t = t.split()
                segment_id = t[0]
                statement = t[1:]

                if "<overlap>" in t:
                    continue

                cleaned_statement = []
                for w in statement:
                    if "<" in w:
                        pass
                    else:
                        cleaned_statement.append(w)

                if len(cleaned_statement) >=8:
                    cleaned_transcript["segment_id"].append(segment_id)
                    cleaned_transcript["transcription"].append(" ".join(cleaned_statement))

            ### Load and Prep Segments ###
            with open(segments, "r") as f: 
                segments = f.readlines()

            cleaned_segments = {"segment_id": [],
                                "path_to_audio": [],
                                "start": [], 
                                "end": []}
            
            segments = [s.split() for s in segments]
            for segment_id, audio_file, start, end in segments:
                cleaned_segments["segment_id"].append(segment_id)
                cleaned_segments["path_to_audio"].append(os.path.join(self.path_to_audios, f"{audio_file}.wav"))
                cleaned_segments["start"].append(float(start))
                cleaned_segments["end"].append(float(end))

            transcripts = pd.DataFrame.from_dict(cleaned_transcript)
            segments = pd.DataFrame.from_dict(cleaned_segments)

            data = pd.merge(transcripts, segments, how="left")
            dataset.append(data)
            
        dataset = pd.concat(dataset)

        ### Will later sample to 16000 Hz so get start and end frame ###
        dataset["start_frame"] = dataset["start"] * self.sr
        dataset["end_frame"] = dataset["end"] * self.sr

        ### Remove any Special Characters from Transcript and Lowecase ###
        dataset["transcription"] = dataset["transcription"].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()

        return dataset


class L2Arctic:
    def __init__(self, dataset_config=DatasetConfig, **kwargs):
        
        self.name = "L2Arctic"
        self.config = dataset_config.dataset_catalog[self.name]
        self.path_to_root = self.config["path_to_data"]
        self.path_to_md = os.path.join(self.path_to_root, "README.md")

    def build_dataset(self):
        ### Grab Speaker info From Markdown ###
        with open(self.path_to_md, "r") as f:
            md = f.readlines()
        start_idx = [idx for (idx, line) in enumerate(md) if "### File summary and speaker information" in line][0]
        end_idx = [idx for (idx, line) in enumerate(md) if "### Phoneme set" in line][0]

        ### Select line after start index and 2 lines before end index and delete empty line after column names ###
        selected_lines = md[start_idx+3:end_idx-2]

        metadata = {"speaker_id": [], 
                    "native_language": [], 
                    "gender": []}

        ### Strip out the start and end empty chars and save ###
        split_lines = [line.strip().split("|")[1:-1] for line in selected_lines]

        for speaker, gender, native_language, _, _ in split_lines:
            metadata["speaker_id"].append(speaker)
            metadata["native_language"].append(native_language)
            metadata["gender"].append(gender)


        metadata = pd.DataFrame.from_dict(metadata)

        transcript_data = {"transcription": [], 
                           "path_to_audio": [],
                           "speaker_id": []}
        
        for speaker in metadata["speaker_id"].unique():
            path_to_speaker_audio = os.path.join(self.path_to_root, speaker, "wav")
            path_to_speaker_transcripts = os.path.join(self.path_to_root, speaker, "transcript")

            for path_to_transcript in os.listdir(path_to_speaker_transcripts):
                filename = path_to_transcript.split(".")[0]
                path_to_audio = f"{filename}.wav"
                path_to_audio = os.path.join(path_to_speaker_audio, path_to_audio)
                if os.path.isfile(path_to_audio):
                    with open(os.path.join(path_to_speaker_transcripts, path_to_transcript)) as f:
                        transcript = (" ".join(f.readlines())).strip()

                    transcript_data["transcription"].append(transcript)
                    transcript_data["path_to_audio"].append(path_to_audio)
                    transcript_data["speaker_id"].append(path_to_audio.split("/")[2])

        transcript_data = pd.DataFrame.from_dict(transcript_data)
        transcript_data["transcription"] = transcript_data["transcription"].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()

        data = pd.merge(transcript_data, metadata, how="left")
        return data


class Mozilla:
    def __init__(self, 
                 dataset_config=DatasetConfig,
                 inference_config=InferenceConfig):
        
        self.name = "MozillaCommonVoice"
        self.config = dataset_config.dataset_catalog[self.name]
        self.path_to_root = self.config["path_to_data"]
        self.accented_subset = self.config["accent_subset"]
        # self.sr = inference_config["sample_rate"]

    def build_dataset(self):
        dataset = load_from_disk(os.path.join(self.path_to_root, self.accented_subset))
        dataset_train = dataset["train"]
        dataset_val = dataset["validation"]
        dataset_test = dataset["test"]

        dataset = concatenate_datasets([dataset_train, dataset_val, dataset_test])
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        print(dataset)
        return dataset
    

@dataclass
class SupportedDatasets:
    """
    Quick class for data implementation sanity check
    """
    supported_dataset: tuple = (Coraal, SpeechAccentArchive, \
                               Edacc, L2Arctic, Mozilla)
    


if __name__ == "__main__":
    data = Mozilla().build_dataset()
    print(data)