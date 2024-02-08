from dataclasses import dataclass

class Config:
    sample_rate: int = 16000
    path_to_pretrained_models: str = "models/"