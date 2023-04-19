from dataclasses import dataclass

@dataclass
class Paths:
    log: str
    data: str

@dataclass
class Files:
    features_data: str
    labels_data: str

@dataclass
class Params:
    epochs: int
    lr: float
    batch_size: int

@dataclass
class SimpleEmbeddingConfig: 
    files: Files
    paths: Paths
    params: Params