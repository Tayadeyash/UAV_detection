from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    num_classes: int = 2  # background + UAV
    image_size: int = 800
    batch_size: int = 4
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    score_threshold: float = 0.35
    nms_iou_threshold: float = 0.50
    num_workers: int = 2
    seed: int = 42

DEFAULT = TrainConfig()
