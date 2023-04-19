from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any

@dataclass
class Adam:
    _target_: str = "torch.optim.adam.Adam"
    params: Any = MISSING
    lr: Any = 0.001
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0
    amsgrad: Any = False