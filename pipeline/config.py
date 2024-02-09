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

    model_config: dict = {"sew": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]}, 
                          "sewd": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},
                          "speech2text": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},
                          "speecht5": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},
                          "unispeech": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},

                          "unispeechsat": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},

                          "wav2vec2": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},
                            
                          "wavlm": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},

                          "whisper": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]},

                          "conformer": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
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

model = SEWForCTC.from_pretrained("patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", cache_dir="models/")
print(sum(p.numel() for p in model.parameters()))
