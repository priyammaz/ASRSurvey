import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from functools import reduce
from audio_datasets import SupportedDatasets
from dataset_loader import BatchPreparation
from config import Config
from accelerate import Accelerator, utils
from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, WhisperForConditionalGeneration, \
                                WhisperProcessor, SpeechT5Processor, SpeechT5ForSpeechToText


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
                 model_config=None):
    
        assert isinstance(dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"

        self.auto_batch_flag = True if batch_size == "auto" else False
        if self.auto_batch_flag:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

        self.num_workers = num_workers
        self.bp = BatchPreparation(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.loader = self.bp.build_dataloader()

        self.model_store = Config.path_to_pretrained_models
        self.sr = Config.sample_rate

        self._init_results()
    
    def _init_results(self):
        self.sew_results = None
        self.sewd_results = None
        self.speech2text_results = None
        self.unispeech_results = None
        self.unispeechsat_results = None
        self.wav2vec2_results = None
        self.conformer_results = None
        self.wavlm_results = None
        self.whisper_results = None

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
                counter = 0 
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
                    batch["transcriptions"] = transcriptions
                    results.append(batch)
                    progress_bar.update(1)

                    counter += 1
                    if counter == 10:
                        break
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
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir="models/")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", cache_dir="models/")

        results = self.distributed_inference(processor=processor, 
                                             model=model, 
                                             forward_pass=self.whisper_forward_pass)

        results = self._flatten_dict_list(results)
        results = pd.DataFrame.from_dict(results)
        results.to_csv("whisper.csv")

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
    ia = InferenceAudios(batch_size="auto", dataset=L2Arctic())
    ia.inference()
        
