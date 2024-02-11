import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from audio_datasets import SupportedDatasets
from dataset_loader import BatchPreparation
from config import InferenceConfig
from accelerate import Accelerator, utils, find_executable_batch_size

class InferenceAudios:
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
                 batch_size=64, 
                 num_workers=8,
                 inference_config=InferenceConfig,
                 only_inference = None, 
                 exclude_inference = None,
                 limit_parameter_size = None):
        
        ### Initialize Dataset and Sanity Checks ###
        self.dataset = dataset()
        assert isinstance(self.dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"
        self.path_to_results_root = self.dataset.config["path_to_results"]
        if not os.path.isdir(self.path_to_results_root):
            os.mkdir(self.path_to_results_root)


        ### Handle Auto BatchSize Search ###
        self.batch_size = batch_size
        
        ### Build DataLoader ###
        self.num_workers = num_workers
        self.bp = BatchPreparation(self.dataset, 
                                   batch_size=self.batch_size, 
                                   num_workers=self.num_workers)

        ### Load in Inference Configs ###
        self.inference_config = inference_config
        self.model_store = self.inference_config.path_to_pretrained_models
        self.sr = self.inference_config.sample_rate

        ### Filter Inference Config to Selected Models ###
        self.only_inference = only_inference
        self.exlude_inference = exclude_inference
        self.limit_parameter_size = limit_parameter_size

        if (self.only_inference is not None) and (self.exlude_inference is not None):
            raise Exception("Either limit model selection with only inference, or remove models with exlude inference")
        
        self.model_catalog = self._limit_model_selection(self.inference_config.model_catalog)

    
    def _limit_model_selection(self, model_catalog):
        if self.only_inference is not None:
            for model in self.model_catalog:
                if model not in self.only_inference:
                    model_catalog.pop(model)
            
        elif self.exlude_inference is not None:
            for model in self.model_catalog:
                if model in self.exlude_inference:
                    model_catalog.pop(model)
            
        if self.limit_parameter_size is not None:
            upper_limit_params = self.limit_parameter_size * 1000000
            for model in self.model_catalog:
                if self.model_catalog[model]["configs"]["params"] > upper_limit_params:
                    model_catalog.pop(model)
    
        print(f"Inferencing on the Models {list(model_catalog.keys())}")
        return model_catalog


    def _flatten_dict_list(self, results):
        final_results = {key:[] for key in results[0].keys()}
        for r in results:
            for key in final_results.keys():
                final_results[key].extend(r[key])
        return final_results
    

    @torch.no_grad()
    def distributed_inference(self, processor, model, forward_pass):
        
        accelerator = Accelerator()

        loader = self.bp.build_dataloader(batch_size=self.batch_size)

        ### Instantiate Accelerator and Prep Objects ###
        prepped_model, prepped_loader = accelerator.prepare(model, loader)
        prepped_model.eval()
        
        ### Iterate through Dataset ###
        progress_bar = tqdm(range(len(prepped_loader)), disable=(not accelerator.is_local_main_process))

        results = []
        for batch in prepped_loader:
            audio = batch.pop("audio")
            
            ### Packaged batch into a dummy list for gather later ###
            batch = [batch]

            ### Prep and Pass Through Model ###
            transcriptions = forward_pass(accelerator=accelerator,
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
        for model in self.model_catalog:
            ### Load Processor, Model and forward_pass ###
            processor_class = self.model_catalog[model]["processor"]
            model_class = self.model_catalog[model]["model"]
            forward_method = self.model_catalog[model]["forward_method"]

            for model_variant in self.model_catalog[model]["configs"]:
                
                ### Grab Checkpoint ID and Name ###
                self.model_id = model_variant["id"]
                chkpt = model_variant["model_config"]

                ### Set Path to Results File ###
                self.path_to_model_results = os.path.join(self.path_to_results_root, f"{self.model_id}.csv")

                if not os.path.isfile(self.path_to_model_results) or (start_from == "scratch"):
                    print(f"Inferencing {self.model_id.upper()} on {self.dataset.name}")

                    ### Load Pre-Trained Model Class and Processor ###
                    processor = processor_class.from_pretrained(chkpt, cache_dir=self.model_store)
                    model = model_class.from_pretrained(chkpt, cache_dir=self.model_store)
            
                    results = self.distributed_inference(processor=processor, 
                                                         model=model, 
                                                         forward_pass=forward_method)

                    results = self._flatten_dict_list(results)
                    results = pd.DataFrame.from_dict(results)
                    results.to_csv(self.path_to_model_results, index=False)

                elif (start_from == "resume"):
                    print(f"Already Inferenced {self.model_id} on {self.dataset.name}")
                    continue


if __name__ == "__main__":
    from audio_datasets import L2Arctic
    ia = InferenceAudios(batch_size=32, dataset=L2Arctic)
    ia.inference()