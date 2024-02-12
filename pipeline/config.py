from forwards import default_forward_pass, speech2text_forward_pass, speecht5_forward_pass, whisper_forward_pass
from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, WhisperForConditionalGeneration, \
                                WhisperProcessor, SpeechT5Processor, SpeechT5ForSpeechToText

class DatasetConfig:
    dataset_catalog: dict = {"CORAAL": {"path_to_data": "data/coraal", 
                                        "path_to_results": "results/coraal"}, 
                             
                             "SpeechAccentArchive": {"path_to_data": "data/speech_accent_archive", 
                                                     "download_file_name": "archive.zip",
                                                     "path_to_results": "results/speech_accent_archive"}, 

                             "EDACC": {"path_to_data": "data/edacc", 
                                       "path_to_results": "results/edacc"}, 

                             "L2Arctic": {"path_to_data": "data/l2arctic", 
                                          "download_file_name": "l2arctic_release_v5.0.zip",
                                          "path_to_results": "results/l2arctic"}, 

                             "MozillaCommonVoice": {"path_to_data": "data/mozilla/", 
                                                    "accent_subset": "accented_mozilla.hf", 
                                                    "path_to_results": "results/mozilla"}}

class InferenceConfig:
    sample_rate: int = 16000

    path_to_pretrained_models: str = "models/"

    model_catalog: dict = {"sew": {"processor": AutoProcessor, 
                                   "model": SEWForCTC, 
                                   "forward_method": default_forward_pass,
                                   "configs": [
                                       
                                      {"model_config": "patrickvonplaten/sew-small-100k-timit", 
                                       "params": 89644350, 
                                       "id": "sew_small",
                                       "batch_size": 16},

                                      {"model_config": "patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", 
                                       "params": 174698814, 
                                       "id": "sew_mid",
                                       "batch_size": 16}

                                  ]}, 

                           "sewd": {"processor": AutoProcessor, 
                                   "model": SEWDForCTC, 
                                   "forward_method": default_forward_pass,
                                   "configs": [
                                       
                                      {"model_config": "asapp/sew-d-tiny-100k-ft-ls100h", 
                                       "params": 24127423, 
                                       "id": "sewd_tiny",
                                       "batch_size": 16},

                                      {"model_config": "asapp/sew-d-mid-k127-400k-ft-ls100h", 
                                       "params": 80389023,
                                       "id": "sewd_mid",
                                       "batch_size": 16},

                                      {"model_config": "asapp/sew-d-base-plus-400k-ft-ls100h", 
                                       "params": 177003711, 
                                       "id": "sewd_base",
                                       "batch_size": 16}
                                  ]},

                           "speech2text": {"processor": Speech2TextProcessor, 
                                  "model": Speech2TextForConditionalGeneration, 
                                  "forward_method": speech2text_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "facebook/s2t-small-librispeech-asr", 
                                       "params": 31335424, 
                                       "id": "s2t_small",
                                       "batch_size": 16},

                                      {"model_config": "facebook/s2t-medium-librispeech-asr", 
                                       "params": 74806272, 
                                       "id": "s2t_medium",
                                       "batch_size": 16},

                                      {"model_config": "facebook/s2t-large-librispeech-asr", 
                                       "params": 275031040, 
                                       "id": "sew_large",
                                       "batch_size": 16}

                                  ]},

                           "speecht5": {"processor": SpeechT5Processor, 
                                  "model": SpeechT5ForSpeechToText, 
                                  "forward_method": speecht5_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "microsoft/speecht5_asr", 
                                       "params": 154588800, 
                                       "id": "speecht5",
                                       "batch_size": 16},

                                  ]},

                           "unispeech": {"processor": AutoProcessor, 
                                  "model": UniSpeechForCTC, 
                                  "forward_method": default_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "patrickvonplaten/unispeech-large-1500h-cv-timit", 
                                       "params": 315470495, 
                                       "id": "unispeech_large",
                                       "batch_size": 16}

                                  ]},

                           "unispeechsat": {"processor": AutoProcessor, 
                                  "model": UniSpeechSatForCTC, 
                                  "forward_method": default_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "microsoft/unispeech-sat-base-100h-libri-ft", 
                                       "params": 94396320, 
                                       "id": "unispeechsat_base",
                                       "batch_size": 16}

                                  ]},

                           "wav2vec2": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ForCTC, 
                                  "forward_method": default_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "facebook/wav2vec2-base-960h", 
                                       "params": 94396320, 
                                       "id": "wav2vec2_base",
                                       "batch_size": 16},

                                      {"model_config": "facebook/wav2vec2-large-960h", 
                                       "params": 315461792, 
                                       "id": "wav2vec2_large",
                                       "batch_size": 16}

                                  ]},

                           "wavlm": {"processor": AutoProcessor, 
                                  "model": WavLMForCTC, 
                                  "forward_method": default_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-base", 
                                       "params": 94405775, 
                                       "id": "wavlm_base",
                                       "batch_size": 16},

                                      {"model_config": "patrickvonplaten/wavlm-libri-clean-100h-large", 
                                       "params": 315484895, 
                                       "id": "wavlm_large",
                                       "batch_size": 16}

                                  ]},

                           "whisper": {"processor": WhisperProcessor, 
                                  "model": WhisperForConditionalGeneration, 
                                  "forward_method": whisper_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "openai/whisper-tiny", 
                                       "params": 37760640, 
                                       "id": "whisper_tiny",
                                       "batch_size": 16},

                                      {"model_config": "openai/whisper-base", 
                                       "params": 72593920, 
                                       "id": "whisper_base",
                                       "batch_size": 16},

                                      {"model_config": "openai/whisper-small", 
                                       "params": 241734912, 
                                       "id": "whisper_small",
                                       "batch_size": 16},

                                      {"model_config": "openai/whisper-medium", 
                                       "params": 763857920, 
                                       "id": "whisper_medium",
                                       "batch_size": 128},

                                      {"model_config": "openai/whisper-large-v2", 
                                       "params": 1543304960, 
                                       "id": "whisper_large",
                                       "batch_size": 8}
                                       

                                  ]},

                           "conformer": {"processor": AutoProcessor, 
                                  "model": Wav2Vec2ConformerForCTC, 
                                  "forward_method": default_forward_pass,
                                  "configs": [
                                      
                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-960h-ft", 
                                       "params": 618591904, 
                                       "id": "conformer_960_rel_large",
                                       "batch_size": 16},

                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-960h-ft", 
                                       "params": 593376928, 
                                       "id": "conformer_960_rope_large",
                                       "batch_size": 16},

                                      {"model_config": "facebook/wav2vec2-conformer-rel-pos-large-100h-ft", 
                                       "params": 618591904, 
                                       "id": "conformer_100_rel_large",
                                       "batch_size": 16},

                                      {"model_config": "facebook/wav2vec2-conformer-rope-large-100h-ft", 
                                       "params": 593376928, 
                                       "id": "conformer_100_rope_large",
                                       "batch_size": 16}
                                       

                                  ]},
                         }
