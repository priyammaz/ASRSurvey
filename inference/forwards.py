import torch 

def default_forward_pass(accelerator,
                         audio,
                         processor,
                         sampling_rate,
                         model):
        
    ### Prepare Inputs and Place in Correct GPU ###
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(model.device)
    logits = model(**inputs).logits

    ### Gather logits across GPU's ###
    logits = accelerator.pad_across_processes(logits, dim=1)
    logits = accelerator.gather(logits)
    predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    ### Decode and Cleanup Transcriptions ###
    transcriptions = processor.batch_decode(predicted_ids)
    transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
    return transcriptions

def whisper_forward_pass(accelerator, 
                         audio, 
                         processor, 
                         sampling_rate,
                         model):
    
    ### Process Input and Place in Correct GPU ###
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(model.device)

    ### Need to Unwrap Model, we need to to .generate() but distributed method only has foward() ###
    unwrapped_model = accelerator.unwrap_model(model)
    generated_ids = unwrapped_model.generate(input_features=inputs, language='en')

    ### Gather Logits Across GPUS ###
    generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
    generated_ids = accelerator.gather(generated_ids)

    ### Decode and Cleanup Transcriptions ###
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
    return transcriptions

def speech2text_forward_pass(accelerator, 
                             audio, 
                             processor, 
                             sampling_rate,
                             model):
        
    ### Process Input and Place in Correct GPU ###
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(model.device)

    ### Need to Unwrap Model, we need to to .generate() but distributed method only has foward() ###
    unwrapped_model = accelerator.unwrap_model(model)
    generated_ids = unwrapped_model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

    ### Gather Logits Across GPUS ###
    generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
    generated_ids = accelerator.gather(generated_ids)

    ### Decode and Cleanup Transcriptions ###
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
    return transcriptions

def speecht5_forward_pass(accelerator, 
                          audio, 
                          processor, 
                          sampling_rate, 
                          model):
    
    ### Process Input and Place in Correct GPU ###
    inputs = processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(model.device)

    ### Need to Unwrap Model, we need to to .generate() but distributed method only has foward() ###
    unwrapped_model = accelerator.unwrap_model(model)
    generated_ids = unwrapped_model.generate(**inputs, max_length=100)

    ### Gather Logits Across GPUS ###
    generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
    generated_ids = accelerator.gather(generated_ids)

    ### Decode and Cleanup Transcriptions ###
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcriptions = [t.replace('[^a-zA-Z\s]', '').lower().strip() for t in transcriptions]
    return transcriptions
