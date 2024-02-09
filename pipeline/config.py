from dataclasses import dataclass

from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, WhisperForConditionalGeneration, \
                                WhisperProcessor, SpeechT5Processor, SpeechT5ForSpeechToText


class Config:
    sample_rate: int = 16000
    path_to_pretrained_models: str = "models/"

    # model_config: dict = {"sew": {
    #     {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
    #     {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft" 174698814}
    #     }, 
    # }

    model_catalog: dict = {"sew": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]}, 

                           "sewd": {"processor": AutoProcessor, 
                                  "model": SEWDForCTC, 
                                  "configs": [
                                      {"model_config": "asapp/sew-d-tiny-100k-ft-ls100h", "params": 89644350},
                                      {"model_config": "asapp/sew-d-mid-k127-400k-ft-ls100h", "params": 174698814},
                                      {"model_config": "asapp/sew-d-base-plus-400k-ft-ls100h", "params": 174698814}
                                  ]},

                           "speech2text": {"processor": Speech2TextProcessor, 
                                  "model": Speech2TextForConditionalGeneration, 
                                  "configs": [
                                      {"model_config": "facebook/s2t-small-librispeech-asr", "params": 89644350},
                                      {"model_config": "facebook/s2t-medium-librispeech-asr", "params": 174698814},
                                      {"model_config": "facebook/s2t-large-librispeech-asr", "params": 174698814}
                                  ]},

                           "speecht5": {"processor": SpeechT5Processor, 
                                  "model": SpeechT5ForSpeechToText, 
                                  "configs": [
                                      {"model_config": "microsoft/speecht5_asr", "params": 89644350},
                                  ]},

                           "unispeech": {"processor": AutoProcessor, 
                                  "model": UniSpeechForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/unispeech-large-1500h-cv-timit", "params": 89644350}
                                  ]},

                           "unispeechsat": {"processor": AutoProcessor, 
                                  "model": UniSpeechSatForCTC, 
                                  "configs": [
                                      {"model_config": "microsoft/unispeech-sat-base-100h-libri-ft", "params": 89644350}
                                  ]},

                           "wav2vec2": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ForCTC, 
                                  "configs": [
                                      {"model_config": "facebook/wav2vec2-base-960h", "params": 89644350},
                                      {"model_config": "facebook/wav2vec2-large-960h", "params": 174698814}
                                  ]},

                            
                           "xlsr": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ForCTC, 
                                  "configs": [
                                      {"model_config": "jonatasgrosman/wav2vec2-large-xlsr-53-english", "params": 89644350},
                                  ]},


                           "wavlm": {"processor": AutoProcessor, 
                                  "model": WavLMForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-base", "params": 89644350},
                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-large", "params": 174698814}
                                  ]},

                           "whisper": {"processor": WhisperProcessor, 
                                  "model": WhisperForConditionalGeneration, 
                                  "configs": [
                                      {"model_config": "openai/whisper-tiny", "params": 89644350},
                                      {"model_config": "openai/whisper-small", "params": 89644350},
                                      {"model_config": "openai/whisper-medium", "params": 89644350},
                                      {"model_config": "openai/whisper-base", "params": 89644350},
                                      {"model_config": "openai/whisper-large-v2", "params": 89644350}
                                  ]},

                           "conformer": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ConformerForCTC, 
                                  "configs": [
                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-960h-ft", "params": 89644350},
                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-960h-ft", "params": 174698814},
                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-100h-ft", "params": 89644350},
                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-100h-ft", "params": 89644350}
                                  ]},
                         }
    

                        #   "sewd": sewd_config,
                        #   "speech2text": speech2text_config, 
                        #   "unispeech": unispeech_config,
                        #   "unispeechsat": unispeechsat_config,
                        #   "wav2vec2": wav2vec2_config,
                        #   "conformer": conformer_config, 
                        #   "wavlm": wavlm_config, 
                        #   "whisper": whisper_config}


catalog = Config.model_catalog

for key, value in catalog.items():
    print(key)
    processor = value["processor"]
    model = value["model"]

    for config in value["configs"]:
        model = model.from_pretrained(config["model_config"], cache_dir="models/")

# print(sum(p.numel() for p in model.parameters()))
