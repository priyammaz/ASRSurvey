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
                 batch_size=4, 
                 device="cuda" if torch.cuda.is_available else "cpu",
                 sew_config="patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", 
                 sewd_config="asapp/sew-d-base-plus-400k-ft-ls100h", 
                 speech2text_config="facebook/s2t-large-librispeech-asr",
                 unispeech_config="patrickvonplaten/unispeech-large-1500h-cv-timit",
                 unispeechsat_config="microsoft/unispeech-sat-base-100h-libri-ft",
                 wav2vec2_config="facebook/wav2vec2-base-960h",
                 conformer_config="facebook/wav2vec2-conformer-rope-large-960h-ft",
                 wavlm_config="patrickvonplaten/wavlm-libri-clean-100h-base-plus", 
                 whisper_config="openai/whisper-tiny"):
    
        assert isinstance(dataset, SupportedDatasets.supported_dataset), "Make sure to use one of the datasets shown in the config"

        self.loader = BatchPreparation(dataset, batch_size=batch_size, num_workers=8).build_dataloader()

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

    def _merge_results(self, results):
        final_results = {key:[] for key in results[0].keys()}
        for r in results:
            for key in final_results.keys():
                final_results[key].extend(r[key])
        final_results = pd.DataFrame.from_dict(final_results)
        return final_results

    @torch.no_grad()
    def distributed_inference(self, processor, model):
        
        ### Instantiate Accelerator and Prep Objects ###
        accelerator = Accelerator()


    @torch.no_grad()
    def inference_sew(self):
        print("Inferencing with SEW")
        processor = AutoProcessor.from_pretrained(self.model_configs["sew"], cache_dir=self.model_store)
        model = SEWForCTC.from_pretrained(self.model_configs["sew"], cache_dir=self.model_store).to(self.device)
        self.sew_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["sew_pred_transcription"] = transcriptions
            self.sew_results.append(batch)

        self.sew_results = self._merge_results(self.sew_results)

    @torch.no_grad()
    def inference_sewd(self):
        print("Inferencing with SEW-D")
        processor = AutoProcessor.from_pretrained(self.model_configs["sewd"], cache_dir=self.model_store)
        model = SEWDForCTC.from_pretrained(self.model_configs["sewd"], cache_dir=self.model_store).to(self.device)
        self.sewd_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["sewd_pred_transcription"] = transcriptions
            self.sewd_results.append(batch)
        
        self.sewd_results = self._merge_results(self.sewd_results)

    @torch.no_grad()
    def inference_speech2text(self):
        print("Inferencing with Speech2Text")
        processor = Speech2TextProcessor.from_pretrained(self.model_configs["speech2text"], cache_dir=self.model_store)
        model = Speech2TextForConditionalGeneration.from_pretrained(self.model_configs["speech2text"], cache_dir=self.model_store).to(self.device)
        self.speech2text_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["speech2text_pred_transcription"] = transcriptions
            self.speech2text_results.append(batch)
        
        self.speech2text_results = self._merge_results(self.speech2text_results)

    @torch.no_grad()
    def inference_unispeech(self):
        print("Inferencing with UniSpeech")
        processor = AutoProcessor.from_pretrained(self.model_configs["unispeech"], cache_dir=self.model_store)
        model = UniSpeechForCTC.from_pretrained(self.model_configs["unispeech"], cache_dir=self.model_store).to(self.device)
        self.unispeech_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["unispeech_pred_transcription"] = transcriptions
            self.unispeech_results.append(batch)

        self.unispeech_results = self._merge_results(self.unispeech_results)
    
    @torch.no_grad()
    def inference_unispeechsat(self):
        print("Inferencing with UniSpeechSAT")
        processor = AutoProcessor.from_pretrained(self.model_configs["unispeechsat"], cache_dir=self.model_store)
        model = UniSpeechSatForCTC.from_pretrained(self.model_configs["unispeechsat"], cache_dir=self.model_store).to(self.device)
        self.unispeechsat_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["unispeechsat_transcription"] = transcriptions
            self.unispeechsat_results.append(batch)

        self.unispeechsat_results = self._merge_results(self.unispeechsat_results)

    @torch.no_grad()
    def inference_wav2vec2(self):
        print("Inferencing with Wav2Vec2")
        processor = AutoProcessor.from_pretrained(self.model_configs["wav2vec2"], cache_dir=self.model_store)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_configs["wav2vec2"], cache_dir=self.model_store).to(self.device)
        self.wav2vec2_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["wav2vec2_pred_transcription"] = transcriptions
            self.wav2vec2_results.append(batch)

        self.wav2vec2_results = self._merge_results(self.wav2vec2_results)

    @torch.no_grad()
    def inference_conformer(self):
        print("Inferencing with Wav2Vec2-Conformer")
        processor = AutoProcessor.from_pretrained(self.model_configs["conformer"], cache_dir=self.model_store)
        model = Wav2Vec2ConformerForCTC.from_pretrained(self.model_configs["conformer"], cache_dir=self.model_store).to(self.device)
        self.conformer_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["conformer_pred_transcription"] = transcriptions
            self.conformer_results.append(batch)

        self.conformer_results = self._merge_results(self.conformer_results)

    @torch.no_grad()
    def inference_wavlm(self):
        print("Inferencing with WavLM")
        processor = AutoProcessor.from_pretrained(self.model_configs["wavlm"], cache_dir=self.model_store)
        model = WavLMForCTC.from_pretrained(self.model_configs["wavlm"], cache_dir=self.model_store).to(self.device)
        self.wavlm_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            batch["wavlm_pred_transcription"] = transcriptions
            self.wavlm_results.append(batch)

        self.wavlm_results = self._merge_results(self.wavlm_results)

    @torch.no_grad()
    def inference_whisper(self):
        print("Inferencing with Whisper")
        processor = WhisperProcessor.from_pretrained(self.model_configs["whisper"], cache_dir=self.model_store)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_configs["whisper"], cache_dir=self.model_store).to(self.device)
        model.config.forced_decoder_ids = None
        self.whisper_results = []
        for batch in tqdm(self.loader):
            audio = batch.pop("audio")
            inputs = processor(audio, sampling_rate=self.sr, return_tensors="pt").input_features.to(self.device)
            generated_ids = model.generate(inputs=inputs)
            print(generated_ids)
            print(generated_ids.shape)
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
            print(transcriptions)
            print(len(transcriptions))
            assaas
            batch["whisper_pred_transcription"] = transcriptions
            self.whisper_results.append(batch)

        self.whisper_results = self._merge_results(self.whisper_results)



    def inference(self, path_to_store):
        # self.inference_sew()
        # self.inference_sewd()
        # self.inference_speech2text()
        # self.inference_unispeech()
        # self.inference_unispeechsat()
        # self.inference_wav2vec2()
        # self.inference_conformer()
        # self.inference_wavlm()
        self.inference_whisper()

        # final_results = [self.sew_results, self.sewd_results, self.speech2text_results, 
        #                  self.unispeech_results, self.unispeechsat_results, self.wav2vec2_results, 
        #                  self.conformer_results, self.wavlm_results, self.whisper_results]

        # ### Remove Outputs from Models not Run ###
        # final_results = [r for r in final_results if r is not None]

        # ### Merge Dataframes Together ###
        # if len(final_results) > 1:
        #     merged_results =  final_results[0]
        #     merge_cols = [col for col in merged_results.columns if "_pred_" not in col]
        #     for result in final_results[1:]:
        #         merged_results = pd.merge(merged_results, result,
        #                                   how="left", left_on=merge_cols,
        #                                   right_on=merge_cols)
        
        # merged_results.to_csv(path_to_store, index=False)
        
    
    



if __name__ == "__main__":
    from audio_datasets import L2Arctic
    ia = InferenceAudios(batch_size=4, dataset=L2Arctic())
    ia.inference(path_to_store="results/l2arctic_testing.csv")
        
