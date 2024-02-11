from config import InferenceConfig

model_catalog = InferenceConfig.model_catalog

for model, info in model_catalog.items():
    model = info["model"]
    processor = info["processor"]
    for chkpts in info["configs"]:
        model_ = model.from_pretrained(chkpts["model_config"], cache_dir="models/")
        processor_ = processor.from_pretrained(chkpts["model_config"], cache_dir="models/")