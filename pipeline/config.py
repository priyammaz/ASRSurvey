from dataclasses import dataclass

from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, WhisperForConditionalGeneration, \
                                WhisperProcessor, SpeechT5Processor, SpeechT5ForSpeechToText


class Config:
    sample_rate: int = 16000
    path_to_pretrained_models: str = "models/"

    model_catalog: dict = {"sew": {"processor": AutoProcessor, 
                                  "model": SEWForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", "params": 89644350},
                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", "params": 174698814}
                                  ]}, 

                           "sewd": {"processor": AutoProcessor, 
                                  "model": SEWDForCTC, 
                                  "configs": [
                                      {"model_config": "asapp/sew-d-tiny-100k-ft-ls100h", "params": 24127423},
                                      {"model_config": "asapp/sew-d-mid-k127-400k-ft-ls100h", "params": 80389023},
                                      {"model_config": "asapp/sew-d-base-plus-400k-ft-ls100h", "params": 177003711}
                                  ]},

                           "speech2text": {"processor": Speech2TextProcessor, 
                                  "model": Speech2TextForConditionalGeneration, 
                                  "configs": [
                                      {"model_config": "facebook/s2t-small-librispeech-asr", "params": 31335424},
                                      {"model_config": "facebook/s2t-medium-librispeech-asr", "params": 74806272},
                                      {"model_config": "facebook/s2t-large-librispeech-asr", "params": 275031040}
                                  ]},

                           "speecht5": {"processor": SpeechT5Processor, 
                                  "model": SpeechT5ForSpeechToText, 
                                  "configs": [
                                      {"model_config": "microsoft/speecht5_asr", "params": 154588800},
                                  ]},

                           "unispeech": {"processor": AutoProcessor, 
                                  "model": UniSpeechForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/unispeech-large-1500h-cv-timit", "params": 315470495}
                                  ]},

                           "unispeechsat": {"processor": AutoProcessor, 
                                  "model": UniSpeechSatForCTC, 
                                  "configs": [
                                      {"model_config": "microsoft/unispeech-sat-base-100h-libri-ft", "params": 94396320}
                                  ]},

                           "wav2vec2": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ForCTC, 
                                  "configs": [
                                      {"model_config": "facebook/wav2vec2-base-960h", "params": 94396320},
                                      {"model_config": "facebook/wav2vec2-large-960h", "params": 315461792}
                                  ]},

                            
                           "xlsr": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ForCTC, 
                                  "configs": [
                                      {"model_config": "jonatasgrosman/wav2vec2-large-xlsr-53-english", "params": 315472545},
                                  ]},


                           "wavlm": {"processor": AutoProcessor, 
                                  "model": WavLMForCTC, 
                                  "configs": [
                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-base", "params": 94405775},
                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-large", "params": 315484895}
                                  ]},

                           "whisper": {"processor": WhisperProcessor, 
                                  "model": WhisperForConditionalGeneration, 
                                  "configs": [
                                      {"model_config": "openai/whisper-tiny", "params": 37760640},
                                      {"model_config": "openai/whisper-small", "params": 241734912},
                                      {"model_config": "openai/whisper-medium", "params": 763857920},
                                      {"model_config": "openai/whisper-base", "params": 72593920},
                                      {"model_config": "openai/whisper-large-v2", "params": 1543304960}
                                  ]},

                           "conformer": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ConformerForCTC, 
                                  "configs": [
                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-960h-ft", "params": 618591904},
                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-960h-ft", "params": 593376928},
                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-100h-ft", "params": 618591904},
                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-100h-ft", "params": 593376928}
                                  ]},
                         }
