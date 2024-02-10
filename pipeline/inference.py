import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from audio_datasets import SupportedDatasets
from dataset_loader import BatchPreparation
from config import InferenceConfig
from accelerate import Accelerator, utils

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
                 batch_size="auto", 
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
        self.auto_batch_flag = True if batch_size == "auto" else False
        if self.auto_batch_flag:
            self.batch_size = 64
        else:
            self.batch_size = batch_size
        
        ### Build DataLoader ###
        self.num_workers = num_workers
        self.bp = BatchPreparation(self.dataset, 
                                   batch_size=self.batch_size, 
                                   num_workers=self.num_workers)
        self.loader = self.bp.build_dataloader()

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
    
    def default_forward_pass(self,
                             accelerator,
                             audio,
                             processor,
                             model):
        
        ### Prepare Inputs and Place in Correct GPU ###
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
        logits = model(**inputs).logits

        ### Gather logits across GPU's ###
        logits = accelerator.pad_across_processes(logits, dim=1)
        logits = accelerator.gather(logits)
        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        ### Decode and Cleanup Transcriptions ###
        transcriptions = processor.batch_decode(predicted_ids)
        transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
        return transcriptions
    
    def whisper_forward_pass(self,
                             accelerator, 
                             audio, 
                             processor, 
                             model):
        
        ### Process Input and Place in Correct GPU ###
        inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt").input_features.to(model.device)

        ### Need to Unwrap Model, we need to to .generate() but distributed method only has foward() ###
        unwrapped_model = accelerator.unwrap_model(model)
        generated_ids = unwrapped_model.generate(inputs=inputs)

        ### Gather Logits Across GPUS ###
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
        generated_ids = accelerator.gather(generated_ids)

        ### Decode and Cleanup Transcriptions ###
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
        return transcriptions



    @torch.no_grad()
    def distributed_inference(self, 
                              processor, 
                              model, 
                              forward_pass):
        
        ### Instantiate Accelerator and Prep Objects ###
        accelerator = Accelerator()
        model, loader = accelerator.prepare(model, self.loader)
        model.eval()
        
        while True:
            try:
                ### Iterate through Dataset ###
                progress_bar = tqdm(range(len(loader)))

                results = []
                for batch in loader:
                    audio = batch.pop("audio")
                    
                    ### Packaged batch into a dummy list for gather later ###
                    batch = [batch]

                    ### Prep and Pass Through Model ###
                    transcriptions = forward_pass(accelerator=accelerator,
                                                audio=audio, 
                                                processor=processor,
                                                model=model)
                
                    ### Gather Batch Metadata ###
                    batch = utils.gather_object(batch)
                    batch = self._flatten_dict_list(batch)

                    ### Add Transcriptions to Batch ###
                    batch[f"{self.model_id}_transcriptions"] = transcriptions
                    results.append(batch)
                    progress_bar.update(1)

                return results

            except torch.cuda.OutOfMemoryError as e:
                updated_batch_size = self.batch_size // 2

                if (self.auto_batch_flag) and (updated_batch_size > 1) and accelerator.is_local_main_process:
                    print(f"Reducing Batch Size from {self.batch_size} to {updated_batch_size}")
                    self.batch_size = updated_batch_size
                    loader = self.bp.update_batch_size(new_batch_size=self.batch_size)
                    loader = accelerator.prepare(loader)
                    continue
                else:
                    print("Failed Inference, Not Enough Memory for Model")
                    break
            
            except Exception as e:
                print(e.message)
                break
        
        
    def inference(self):
        for model in self.model_catalog:
            ### Check for Previous Inference Model/Data Combo ###
            self.model_id = self.model_catalog[model]["configs"]["id"]
            self.path_to_model_results = os.path.join(self.path_to_results_root, f"{self.model_id}.csv")

            if not os.path.isfile(self.path_to_model_results):
                print(f"Inferencing {self.model_id.upper()} on {self.dataset.name}")
                ### Grab Checkpoint Name ###
                chkpt = self.model_catalog[model]["configs"]["model_config"]

                ### Load Processor and Model ###
                processor = self.model_catalog[model]["processor"].from_pretrained(chkpt, cache_dir=self.model_store)
                model = self.model_catalog[model]["model"].from_pretrained(chkpt, cache_dir=self.model_store)

                results = self.distributed_inference(processor=processor, 
                                                    model=model, 
                                                    forward_pass=self.whisper_forward_pass)

                results = self._flatten_dict_list(results)
                results = pd.DataFrame.from_dict(results)
                results.to_csv(self.path_to_model_results, index=False)

            else:
                print(f"Already Inferenced {self.model_id} on {self.dataset.name}")
                continue

        # processor = AutoProcessor.from_pretrained("asapp/sew-d-base-plus-400k-ft-ls100h", cache_dir="models/")
        # model = SEWDForCTC.from_pretrained("asapp/sew-d-base-plus-400k-ft-ls100h", cache_dir="models/")

        # results = self.distributed_inference(processor=processor, 
        #                                      model=model, 
        #                                      forward_pass=self.default_forward_pass)
 
        # results = self._flatten_dict_list(results)
        # results = pd.DataFrame.from_dict(results)
        # results.to_csv("sewd.csv")
            


if __name__ == "__main__":
    from audio_datasets import L2Arctic
    ia = InferenceAudios(batch_size="auto", dataset=L2Arctic)
    # ia.inference()
        
