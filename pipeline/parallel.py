from audio_datasets import L2Arctic
from dataset_loader import BatchPreparation
from accelerate import Accelerator
from accelerate.utils import pad_across_processes, gather_object
import torch
from tqdm import tqdm
from transformers import AutoProcessor, SEWForCTC, SEWDForCTC, Speech2TextProcessor, Speech2TextForConditionalGeneration, \
                            UniSpeechForCTC, UniSpeechSatForCTC, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, WavLMForCTC, WhisperForConditionalGeneration, \
                                WhisperProcessor

l = L2Arctic()
bp = BatchPreparation(l, batch_size=4, num_workers=4)
loader = bp.build_dataloader()

accelerator = Accelerator()

processor = AutoProcessor.from_pretrained("patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", cache_dir="models/")
model = SEWForCTC.from_pretrained("patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft", cache_dir="models/")

model, loader = accelerator.prepare(model, loader)
model.eval()

progress_bar = tqdm(range(len(loader)))

results = []
counter = 0
for batch in loader:
    audio = batch.pop("audio")
    batch = [batch]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():

        logits = model(**inputs).logits
        logits = accelerator.pad_across_processes(
                logits, dim=1)

        logits = accelerator.gather(logits)
        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    transcriptions = processor.batch_decode(predicted_ids)
    transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
    batch = gather_object(batch)

    if accelerator.is_local_main_process:
        print(batch)
        print(transcriptions)
    batch["sew_pred_transcription"] = transcriptions
    
    results.append(batch)
    progress_bar.update(1)
    
    counter += 1
    if counter == 5:
        break

final_results = {key:[] for key in results[0].keys()}
for r in results:
    for key in final_results.keys():
        final_results[key].extend(r[key])

print(final_results)
for key in final_results.keys():
    print(key, len(final_results[key]))
import pandas as pd
final_results = pd.DataFrame.from_dict(final_results)
final_results.to_csv("sample_out.csv")
