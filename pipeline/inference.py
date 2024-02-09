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
                                WhisperProcessor


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
                 batch_size=128, 
                 device="cuda" if torch.cuda.is_available else "cpu",
                 sew_config="patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", 
                 sewd_config="asapp/sew-d-base-plus-400k-ft-ls100h", 
                 speech2text_config="facebook/s2t-large-librispeech-asr",
                 unispeech_config="patrickvonplaten/unispeech-large-1500h-cv-timit",
                 unispeechsat_config="microsoft/unispeech-sat-base-100h-libri-ft",
                 wav2vec2_config="facebook/wav2vec2-base-960h",
                 conformer_config="facebook/wav2vec2-conformer-rope-large-960h-ft",
                 wavlm_config="patrickvonplaten/wavlm-libri-clean-100h-base-plus", 
                 whisper_config="openai/whisper-medium"):
    
        assert isinstance(dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"

        self.loader = BatchPreparation(dataset, batch_size=batch_size, num_workers=4).build_dataloader()

        self.device = device
        self.model_store = Config.path_to_pretrained_models
        self.sr = Config.sample_rate
        self.model_configs = {"sew": sew_config, 
                              "sewd": sewd_config,
                              "speech2text": speech2text_config, 
                              "unispeech": unispeech_config,
                              "unispeechsat": unispeechsat_config,
                              "wav2vec2": wav2vec2_config,
                              "conformer": conformer_config, 
                              "wavlm": wavlm_config, 
                              "whisper": whisper_config}

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
        
        ### Prepare Inputs and Pass to Model ###
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        logits = model(**inputs).logits

        ### Gather logits across GPU's ###
        logits = accelerator.pad_across_processes(logits, dim=1)
        logits = accelerator.gather(logits)
        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        ### Decode and Cleanup Transcriptions ###
        transcriptions = processor.batch_decode(predicted_ids)
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
            batch["transcriptions"] = transcriptions
            results.append(batch)
            progress_bar.update(1)
        
        return results

    
    def inference(self):
        processor = AutoProcessor.from_pretrained("patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", cache_dir="models/")
        model = SEWForCTC.from_pretrained("patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", cache_dir="models/")

        results = self.distributed_inference(processor=processor, 
                                             model=model, 
                                             forward_pass=self.default_forward_pass)
        # for batch in self.loader:
        #     audio = batch.pop("audio")
        #     inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        #     print(inputs)
        #     break
        results = self._flatten_dict_list(results)
        results = pd.DataFrame.from_dict(results)
        results.to_csv("hello.csv")
            


if __name__ == "__main__":
    from audio_datasets import L2Arctic
    ia = InferenceAudios(batch_size=64, dataset=L2Arctic())
    ia.inference()
        
