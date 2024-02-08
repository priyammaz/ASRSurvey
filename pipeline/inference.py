import pandas as pd
import numpy as np
import librosa
import os
import torch
import transformers
from tqdm import tqdm
from dataset_builder import Coraal, SpeechAccentArchive, Edacc, L2Arctic, Mozilla
from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, \
                                WhisperForConditionalGeneration