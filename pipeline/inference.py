import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from audio_datasets import SupportedDatasets
from dataset_loader import BatchPreparation
from config import InferenceConfig
from accelerate import Accelerator, utils

class DataParallelInference:
    """
    Class to inference a supported dataset on a wide variety of ASR Models on the Huggingface Platform.
    The models we will be exploring are:

    Wav2Vec2: https://huggingface.co/docs/transformers/model_doc/wav2vec2
    Wav2VecConformer: https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer
    SEW: https://huggingface.co/docs/transformers/model_doc/sew
    SEWD: https://huggingface.co/docs/transformers/model_doc/sew-d
    Speech2Text: https://huggingface.co/docs/transformers/model_doc/speech_to_text
    UniSpeech: https://huggingface.co/docs/transformers/model_doc/unispeech
    UniSpeechSAT: https://huggingface.co/docs/transformers/model_doc/unispeech-sat
    WavLM: https://huggingface.co/docs/transformers/model_doc/wavlm
    Whisper: https://huggingface.co/docs/transformers/model_doc/whisper

    """
    def __init__(self, 
                 dataset,
                 model_id,
                 num_workers=8,
                 inference_config=InferenceConfig):
        
        ### Initialize Accelerator Class ###
        self.accelerator = Accelerator()

        ### Initialize Dataset and Sanity Checks ###
        self.dataset = dataset()
        if self.accelerator.is_main_process:
            assert isinstance(self.dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"

        self.path_to_results_root = self.dataset.config["path_to_results"]
        if self.accelerator.is_main_process:
            if not os.path.isdir(self.path_to_results_root):
                os.mkdir(self.path_to_results_root)
        
        ### Build DataLoader ###
        self.bp = BatchPreparation(self.dataset, 
                                   num_workers=num_workers)

        ### Load in Inference Configs ###
        self.inference_config = inference_config
        self.model_store = self.inference_config.path_to_pretrained_models
        self.sr = self.inference_config.sample_rate

        ### Select Model Configuration ###
        self.model_id = model_id
        self.model_config = self._model_selection(self.inference_config.model_catalog)

    def _model_selection(self, model_catalog):
        found_model = False
        for model in model_catalog:
            for config in model_catalog[model]["configs"]:
                if config["id"] == self.model_id:
                    self.chkpt = config["model_config"]
                    self.batch_size = config["batch_size"]
                    self.processor_class = model_catalog[model]["processor"]
                    self.model_class = model_catalog[model]["model"]
                    self.forward_method = model_catalog[model]["forward_method"]
                    found_model = True

       
        if self.accelerator.is_main_process and not found_model:
            raise Exception(f"Modlel ID {self.model_id} not available in InferenceConfig.model_catalog!!!")

    def _flatten_dict_list(self, results):
        final_results = {key:[] for key in results[0].keys()}
        for r in results:
            for key in final_results.keys():
                final_results[key].extend(r[key])
        return final_results

    @torch.no_grad()
    def distributed_inference(self, batch_size, processor, model, forward_pass):

        if self.accelerator.is_main_process:
            print(f"Inferencing {self.model_id.upper()} on {self.dataset.name}")

        loader = self.bp.build_dataloader(batch_size=batch_size)

        ### Instantiate Accelerator and Prep Objects ###
        prepped_model, prepped_loader = self.accelerator.prepare(model, loader)
        prepped_model.eval()
        
        ### Iterate through Dataset ###
        progress_bar = tqdm(range(len(prepped_loader)), disable=(not self.accelerator.is_local_main_process))

        results = []
        for batch in prepped_loader:
            audio = batch.pop("audio")
            
            ### Packaged batch into a dummy list for gather later ###
            batch = [batch]

            ### Prep and Pass Through Model ###
            transcriptions = forward_pass(accelerator=self.accelerator,
                                          audio=audio, 
                                          processor=processor,
                                          sampling_rate=self.sr,
                                          model=prepped_model)
        
            ### Gather Batch Metadata ###
            batch = utils.gather_object(batch)
            batch = self._flatten_dict_list(batch)

            ### Add Transcriptions to Batch ###
            batch[f"{self.model_id}_transcriptions"] = transcriptions
            results.append(batch)
            progress_bar.update(1)

        return results
        
    def inference(self, start_from="resume"):

        ### Set Path to Results File ###
        self.path_to_model_results = os.path.join(self.path_to_results_root, f"{self.model_id}.csv")

        if not os.path.isfile(self.path_to_model_results) or (start_from == "scratch"):

            ### Load Pre-Trained Model Class and Processor ###
            processor = self.processor_class.from_pretrained(self.chkpt, cache_dir=self.model_store)
            model = self.model_class.from_pretrained(self.chkpt, cache_dir=self.model_store)
    
            results = self.distributed_inference(batch_size=self.batch_size,
                                                    processor=processor, 
                                                    model=model, 
                                                    forward_pass=self.forward_method)

            if results is not None:
                results = self._flatten_dict_list(results)
                results = pd.DataFrame.from_dict(results)
                results.to_csv(self.path_to_model_results, index=False)

        elif (start_from == "resume"):
            print(f"Already Inferenced {self.model_id} on {self.dataset.name}")

        ### Clear Memory for Next Iteration ###        
        self.accelerator.free_memory()

if __name__ == "__main__":
    from audio_datasets import L2Arctic

    ia = DataParallelInference(dataset=L2Arctic, model_id="sew_small")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

    ia = DataParallelInference(dataset=L2Arctic, model_id="whisper_medium")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

    ia = DataParallelInference(dataset=L2Arctic, model_id="whisper_medium")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

    ia = DataParallelInference(dataset=L2Arctic, model_id="sew_small")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

    ia = DataParallelInference(dataset=L2Arctic, model_id="whisper_medium")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

    ia = DataParallelInference(dataset=L2Arctic, model_id="sew_small")
    try:
        ia.inference()
    except:
        ia.accelerator.free_memory()
        print("failed")

