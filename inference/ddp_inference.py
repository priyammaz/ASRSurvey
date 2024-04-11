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
    
    Distributed Data Parallel Inferencing class for ASR.

    """
    def __init__(self, 
                 dataset,
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
        self.bp = BatchPreparation(self.dataset, 
                                   num_workers=num_workers)

        ### Load in Inference Configs ###
        self.inference_config = inference_config
        self.model_catalog = self.inference_config.model_catalog
        self.model_store = self.inference_config.path_to_pretrained_models
        self.sr = self.inference_config.sample_rate

    def _model_parts_selection(self):
        found_model = False
        self.chkpt = None
        self.processor_class = None
        self.model_class = None
        self.forward_method = None

        for model in self.model_catalog:
            for config in self.model_catalog[model]["configs"]:
                if config["id"] == self.model_id:
                    self.chkpt = config["model_config"]
                    self.processor_class = self.model_catalog[model]["processor"]
                    self.model_class = self.model_catalog[model]["model"]
                    self.forward_method = self.model_catalog[model]["forward_method"]
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
    def inference(self, model_id, batch_size, start_from="resume"):
        
        ### Define Inference Parameters ###
        self.batch_size = batch_size
        self.model_id = model_id

        ### Grab Model Inference Parts ###
        self._model_parts_selection()

        try:
            if self.chkpt is not None:
                ### Set Path to Results File ###
                self.path_to_model_results = os.path.join(self.path_to_results_root, f"{self.model_id}.csv")

                if not os.path.isfile(self.path_to_model_results) or (start_from == "scratch"):

                    ### Load Pre-Trained Model Class and Processor ###
                    processor = self.processor_class.from_pretrained(self.chkpt, cache_dir=self.model_store)
                    model = self.model_class.from_pretrained(self.chkpt, cache_dir=self.model_store)

                    loader = self.bp.build_dataloader(batch_size=batch_size)

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
                    if self.accelerator.is_main_process:
                        print(f"Already Inferenced {self.model_id} on {self.dataset.name}")
                

            else:
                raise KeyError(f"Cant Find Model {self.model_id} in InferenceConfig.model_catalog")
            
        except Exception as error:
            model = model.to("cpu")
            del model, processor, loader 
            self.accelerator.free_memory()
            torch.cuda.empty_cache()
            gc.collect()
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                print("Error:", type(error).__name__)

            raise error


class DDPMultiModelsInference(DDPInference):
    def __init__(self, 
                 dataset, 
                 only_inference=None, 
                 exclude_inference=None,
                 limit_parameter_size=None,
                 model_ids=None, 
                 accelerator=None,
                 num_workers=4,
                 inference_config=InferenceConfig):
        
        ### Initialize Accelerator ###
        self.accelerator = accelerator if accelerator is not None else Accelerator()

        ### Initialize DDPInference Class
        super().__init__(dataset=dataset, 
                         num_workers=num_workers, 
                         accelerator=accelerator, 
                         inference_config=inference_config)
 

        ### Filter Inference Config to Selected Models ###
        self.inference_config = inference_config
        self.model_catalog = self.inference_config.model_catalog
        self.model_ids = model_ids
        self.only_inference = only_inference
        self.exclude_inference = exclude_inference
        self.limit_parameter_size = limit_parameter_size

        if self.model_ids is not None:
            self.only_inference = None
            self.exclude_inference = None
            self.limit_parameter_size = None

        else:
            if (self.only_inference is not None) and (self.exlude_inference is not None):
                raise Exception("Either limit model selection with only inference, or remove models with exlude inference")
        
        self._filter_model_selection()

    def _filter_model_selection(self):

        if isinstance(self.only_inference, str):
            self.only_inference = [self.only_inference]
        if isinstance(self.exclude_inference, str):
            self.exclude_inference = [self.exclude_inference]
        if isinstance(self.model_ids, str):
            self.model_ids = [self.model_ids]

        if self.only_inference is not None:
            for model in list(self.model_catalog):
                if model not in self.only_inference:
                    self.model_catalog.pop(model)

        elif self.exclude_inference is not None:
            for model in list(self.model_catalog):
                if model in self.exclude_inference:
                    self.model_catalog.pop(model)

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

            for model in list(self.model_catalog):
                for idx, variant in enumerate(self.model_catalog[model]["configs"]):
                    if variant["params"] > upper_limit_params:
                        self.model_catalog[model]["configs"].pop(idx)

        self.selected_models = []
        self.starting_batch_size = []
        for model in self.model_catalog:
            for variants in self.model_catalog[model]["configs"]:

                if self.model_ids is not None:
                    if variants["id"] in self.model_ids:
                        self.selected_models.append(variants["id"])
                        self.starting_batch_size.append(variants["batch_size"])
                else:
                    self.selected_models.append(variants["id"])
                    self.starting_batch_size.append(variants["batch_size"])

        if self.accelerator.is_main_process:
            assert(len(self.selected_models) > 0), "Print No Models Selected, Check"
            print(f"Inferencing on {len(self.selected_models)} Model(s), {self.selected_models}")


    def multiinference(self, batch_size="auto", starting_batch_size=None, start_from="resume"):
        
        self.auto_batch_size_flag = True if batch_size == "auto" else False
        if self.auto_batch_size_flag:
            if starting_batch_size is None:
                inference_list = [(self.selected_models[i], self.starting_batch_size[i]) for i in range(len(self.selected_models))]
            else:
                inference_list = [(self.selected_models[i], starting_batch_size) for i in range(len(self.selected_models))]
            
        ### Loop Through All Model Ids  ###
        for model_id, batch_size in inference_list:
            if self.accelerator.is_main_process:
                print(f"Inferencing {model_id.upper()} on {self.dataset.name} with Batch Size {batch_size}")

            while True:

                try:
                    self.inference(model_id=model_id, 
                                  batch_size=batch_size, 
                                  start_from=start_from)
                    
                    break
                
                except torch.cuda.OutOfMemoryError:

                    if self.auto_batch_size_flag:
                        reduced_batch_size = batch_size // 2

                        if reduced_batch_size >= 1:
                            if self.accelerator.is_main_process:
                                print(f"Reducing Batch Size from {batch_size} to {batch_size//2}")
                            batch_size = reduced_batch_size
                            continue
                        else:
                            print("Not enough Memory for a Batch size of 1, use limit_parameter_size to limit larger models from GPU Inference")
                            break
                    else:
                        break

                except Exception as error:
                    if self.accelerator.is_main_process:
                        print("Error:", type(error).__name__, error)
                        print("Skipping to Next Model!!!")
                    break
        


if __name__ == "__main__":
    from audio_datasets import SpeechAccentArchive, L2Arctic, Mozilla, Coraal, Edacc, SantaBarbara
    accelerator = Accelerator()

    # ip = DDPMultiModelsInference(dataset=Mozilla, model_ids=["sew_small"])
    # ip.multiinference()
    
    # ip = DDPMultiModelsInference(dataset=Mozilla)
    # ip.multiinference()

    ip = DDPMultiModelsInference(dataset=SantaBarbara)
    ip.multiinference()
    
