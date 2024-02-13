import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from audio_datasets import SupportedDatasets
from dataset_loader import BatchPreparation
from config import InferenceConfig
from accelerate import Accelerator, utils
import gc

class DDPInference:
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
                 batch_size,
                 num_workers=8,
                 accelerator=None,
                 inference_config=InferenceConfig):
        
        ### Initialize Accelerator Class ###
        self.accelerator = accelerator if accelerator is not None else Accelerator()

        ### Initialize Dataset and Sanity Checks ###
        self.dataset = dataset()
        if self.accelerator.is_main_process:
            assert isinstance(self.dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"

        self.path_to_results_root = self.dataset.config["path_to_results"]
        if self.accelerator.is_main_process:
            if not os.path.isdir(self.path_to_results_root):
                os.mkdir(self.path_to_results_root)
        
        ### Build DataLoader ###
        self.batch_size = batch_size
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
        self.chkpt = None
        self.processor_class = None
        self.model_class = None
        self.forward_method = None

        for model in model_catalog:
            for config in model_catalog[model]["configs"]:
                if config["id"] == self.model_id:
                    self.chkpt = config["model_config"]
                    self.processor_class = model_catalog[model]["processor"]
                    self.model_class = model_catalog[model]["model"]
                    self.forward_method = model_catalog[model]["forward_method"]
                    found_model = True
        
        if not found_model:
            raise KeyError(f"Could not find model {self.model_id} in Model Catalog")

    def _flatten_dict_list(self, results):
        final_results = {key:[] for key in results[0].keys()}
        for r in results:
            for key in final_results.keys():
                final_results[key].extend(r[key])
        return final_results

    @torch.inference_mode()
    def distributed_inference(self, processor, model, loader, forward_pass):

        if self.accelerator.is_main_process:
            print(f"Inferencing {self.model_id.upper()} on {self.dataset.name}")

        ### Iterate through Dataset ###
        progress_bar = tqdm(range(len(loader)), disable=(not self.accelerator.is_local_main_process))

        results = []
        for batch in loader:
            audio = batch.pop("audio")
            
            ### Packaged batch into a dummy list for gather later ###
            batch = [batch]

            ### Prep and Pass Through Model ###
            transcriptions = forward_pass(accelerator=self.accelerator,
                                          audio=audio, 
                                          processor=processor,
                                          sampling_rate=self.sr,
                                          model=model)
        
            ### Gather Batch Metadata ###
            batch = utils.gather_object(batch)
            batch = self._flatten_dict_list(batch)

            ### Add Transcriptions to Batch ###
            batch[f"{self.model_id}_transcriptions"] = transcriptions
            results.append(batch)
            progress_bar.update(1)

        return results
    
    @torch.inference_mode()
    def inference(self, start_from="resume"):
        
        try:
            if self.chkpt is not None:
                ### Set Path to Results File ###
                self.path_to_model_results = os.path.join(self.path_to_results_root, f"{self.model_id}.csv")

                if not os.path.isfile(self.path_to_model_results) or (start_from == "scratch"):

                    ### Load Pre-Trained Model Class and Processor ###
                    processor = self.processor_class.from_pretrained(self.chkpt, cache_dir=self.model_store)
                    model = self.model_class.from_pretrained(self.chkpt, cache_dir=self.model_store)

                    loader = self.bp.build_dataloader(batch_size=self.batch_size)

                    ### Instantiate Accelerator and Prep Objects ###
                    model, loader = self.accelerator.prepare(model, loader)

                    results = self.distributed_inference(processor=processor, 
                                                         model=model, 
                                                         loader=loader,
                                                         forward_pass=self.forward_method)

                    if results is not None:
                        results = self._flatten_dict_list(results)
                        results = pd.DataFrame.from_dict(results)

                        if self.accelerator.is_main_process:
                            results.to_csv(self.path_to_model_results, index=False)
                    
                    model = model.to("cpu")
                    del model, processor, loader
                    self.accelerator.free_memory()
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.accelerator.wait_for_everyone()
                    

                elif (start_from == "resume"):
                    print(f"Already Inferenced {self.model_id} on {self.dataset.name}")
                

            else:
                raise KeyError(f"Cant Find Model {self.model_id} in InferenceConfig.model_catalog")
            
        except Exception as e:
            model = model.to("cpu")
            del model, processor, loader 
            self.accelerator.free_memory()
            torch.cuda.empty_cache()
            gc.collect()
            self.accelerator.wait_for_everyone()


            if self.accelerator.is_main_process:
                print("Cleared Memory")
                print("Error:", type(e).__name__)

            raise e


class InferencePipeline:
    def __init__(self, 
                 accelerator,
                 dataset, 
                 only_inference=None, 
                 exclude_inference=None,
                 limit_parameter_size=None,
                 inference_config=InferenceConfig):
        
        ### Store Dataset Instance ###
        self.dataset = dataset 

        ### Filter Inference Config to Selected Models ###
        self.inference_config = inference_config
        self.only_inference = only_inference
        self.exlude_inference = exclude_inference
        self.limit_parameter_size = limit_parameter_size

        if (self.only_inference is not None) and (self.exlude_inference is not None):
            raise Exception("Either limit model selection with only inference, or remove models with exlude inference")

        self.model_catalog = self._limit_model_selection(self.inference_config.model_catalog)


    def _limit_model_selection(self, model_catalog):
        if self.only_inference is not None:
            for model in list(model_catalog):
                if model not in self.only_inference:
                    model_catalog.pop(model)

        elif self.exlude_inference is not None:
            for model in list(model_catalog):
                if model in self.exlude_inference:
                    model_catalog.pop(model)

        if self.limit_parameter_size is not None:

            factor_map = {"M": 1000000, 
                          "B": 1000000000}

            if isinstance(self.limit_parameter_size, str):
                number, factor = float(self.limit_parameter_size[:-1]), self.limit_parameter_size[-1]
                assert factor in factor_map.keys(), "Make sure input is integer or in string form i.e. 100M for 100 Million or 1B for 1 Billion"
                upper_limit_params = number * factor_map[factor]

            elif isinstance(self.limit_parameter_size, int): 
                upper_limit_params = self.limit_parameter_size

            else:
                raise Exception("Make sure Input is in correct format: Integer Input (2000000), or String Input (20M -> 20 million, 1B -> 1 Billion)")

            for model in list(model_catalog):
                for idx, variant in enumerate(model_catalog[model]["configs"]):
                    if variant["params"] > upper_limit_params:
                        model_catalog[model]["configs"].pop(idx)

        return model_catalog



if __name__ == "__main__":
    from audio_datasets import SpeechAccentArchive, L2Arctic
    accelerator = Accelerator()

    ip = InferencePipeline()

    # batch_size = 128
    # while True:
    #     ia = DDPInference(accelerator=accelerator,
    #                       dataset=SpeechAccentArchive, 
    #                       model_id="whisper_medium", batch_size=batch_size)
    #     try:
    #         ia.inference()
    #         print("ITMADEITTT!!!!!")
    #         break
    #     except Exception as e:
    #         print(f"Reducing batch size from {batch_size} to {batch_size//2}")
    #         batch_size = batch_size // 2
    #         print(e)
    #         continue
    
    # batch_size = 128
    # while True:
    #     ia = DataParallelInference(dataset=SpeechAccentArchive, model_id="whisper_large", batch_size=batch_size)
    #     try:
    #         ia.inference()
    #         break
    #     except Exception as e:
    #         print(f"Reducing batch size from {batch_size} to {batch_size//2}")
    #         batch_size = batch_size // 2
    #         print(e)
    #         continue


    # batch_size = 128
    # while True:
    #     ia = DataParallelInference(dataset=SpeechAccentArchive, model_id="conformer_960_rel_large", batch_size=batch_size)
    #     try:
    #         ia.inference()
    #         break
    #     except Exception as e:
    #         print(f"Reducing batch size from {batch_size} to {batch_size//2}")
    #         batch_size = batch_size // 2
    #         print(e)
    #         continue
    
    # batch_size = 128
    # while True:
    #     ia = DataParallelInference(dataset=SpeechAccentArchive, model_id="conformer_960_rope_large", batch_size=batch_size)
    #     try:
    #         ia.inference()
    #         break
    #     except Exception as e:
    #         print(f"Reducing batch size from {batch_size} to {batch_size//2}")
    #         batch_size = batch_size // 2
    #         print(e)
    #         continue








