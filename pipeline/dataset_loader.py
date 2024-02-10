import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import Config
from audio_datasets import Coraal, SpeechAccentArchive, Edacc, L2Arctic, Mozilla

SR = Config.sample_rate

class AudioLoader(Dataset):
    def __init__(self, dataset, load_array=True):
        self.dataset = dataset
        self.load_array = load_array
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.load_array:
            row = dict(self.dataset.loc[idx, :])
            audio = librosa.load(row["path_to_audio"], sr=SR)[0]
            row["audio"] = audio
        else:
            row = self.dataset[idx]

        return row

class BatchPreparation:

    """

    For the most efficient inference, we need to batch-predict data but
    we have 3 options for the current datasets:

    (1) Pre-Batch Data:
    CORAAL and EDACC have long form audio along with the start and end times of phrases, 
    we can preload all the audio into batches of tuples containing audio, transcript and metadata
    and then pass to our Audioloader

    (2) Realtime Loading:
    Speech Accent Archive and L2 Arctic contain audio clips cut up into shorter segments and 
    we can just load them as we grab batches from our dataset with the AudioLoader

    (3) Huggingface Datset
    Mozilla Common Voice is a huggingface dataset so we can just use regular PyTorch Dataloaders

    """

    def __init__(self, 
                 dataset, 
                 batch_size=128, 
                 num_workers=0):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build_dataloader(self, limit_audio_num=None):

        if isinstance(self.dataset, (Coraal, Edacc)):
            dataloader = self.pre_batch_data(limit_audio_num)
        elif isinstance(self.dataset, (SpeechAccentArchive, L2Arctic)):
            dataloader = self.realtime_load()
        elif isinstance(self.dataset, (Mozilla)):
            dataloader = self.huggingface_loader(collate_fn=self._mozilla_collate_function)
        
        return dataloader

    def _collate_function(self, examples):
        
        """

        Convert a list of dictionaries to a dictionary of lists  

        """

        batch_dict = {key: [] for key in examples[0].keys()}
        for e in examples:
            for k in batch_dict.keys():
                batch_dict[k].append(e[k])
        return batch_dict
    
    def _mozilla_collate_function(self, examples):
        
        """

        Mozilla Dataset has the audio array as a subkey of audio, we just extract it out
        and then use default collate function

        """
        for e in examples:
            e["audio"] = e["audio"]["array"]
        return self._collate_function(examples)

    def pre_batch_data(self, limit_audio_num=None):

        """
        Loads all data to memory and then cuts/batches segments for inference 

        """

        ### Build Dataframe of Audio Information ###
        dataset = self.dataset.build_dataset()
        
        assert ("start_frame" in dataset.columns) and ("end_frame" in dataset.columns), "Prebatching is for longform audio with preset-segments"
        if limit_audio_num is not None:
            assert isinstance(limit_audio_num, int), "Limit the number of audios loaded to memory with an integer input for limit_audio_num"

        print("Loading all Audio to Memory")
        audios = dataset.path_to_audio.unique()
        if limit_audio_num is not None:
            audios = audios[:limit_audio_num]

        audio_dict = {}
        for path in tqdm(audios):
            audio_dict[path] = librosa.load(path, sr=SR)[0]
        

        print("Cutting Down Audio to Segments")
        samples = []

        for idx, row in dataset.iterrows():
            try:
                data = dict(row)
                audio_slice = audio_dict[row["path_to_audio"]][int(row["start_frame"]):int(row["end_frame"])]
                data["audio"] = audio_slice
                samples.append(data)
            except:
                continue


        dataset = AudioLoader(samples, load_array=False)
        loader = DataLoader(dataset, 
                            batch_size=self.batch_size, 
                            collate_fn=self._collate_function, 
                            num_workers=self.num_workers)
        
        return loader

    def realtime_load(self):

        """
        Load audios as they come in with super simple Dataset class AudioLoader

        """

        dataset = self.dataset.build_dataset()
        dataset = AudioLoader(dataset)
        loader = DataLoader(dataset, 
                            batch_size=self.batch_size, 
                            collate_fn=self._collate_function,
                            num_workers=self.num_workers)
        
        return loader
    
    def huggingface_loader(self, collate_fn):
        
        """
        Load audios from huggingface datasets, currently only tested for Mozilla

        """

        dataset = self.dataset.build_dataset()
        loader = DataLoader(dataset, 
                            collate_fn=collate_fn,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        
        return loader


if __name__ == "__main__":
    print("CORAAL")
    c = Coraal()
    bp = BatchPreparation(c, batch_size=4)
    loader = bp.build_dataloader(limit_audio_num=10)

    for data in loader:
        print(data)
        break

    print("EDACC")
    e = Edacc()
    bp = BatchPreparation(e, batch_size=4)
    loader = bp.build_dataloader(limit_audio_num=10)

    for data in loader:
        print(data)
        break

    print("L2Arctic")
    l = L2Arctic()
    bp = BatchPreparation(l, batch_size=4)
    loader = bp.build_dataloader()

    for data in loader:
        print(data)
        break

    print("SAA")
    s = SpeechAccentArchive()
    bp = BatchPreparation(s, batch_size=4)
    loader = bp.build_dataloader()

    for data in loader:
        print(data)
        break

    print("Mozilla")
    m = Mozilla()
    bp = BatchPreparation(m, batch_size=4)
    loader = bp.build_dataloader()

    for data in loader:
        print(data)
        break
