from dataclasses import dataclass
from typing import Any, Dict

from konductor.data import get_dataset_properties
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig

from .LSTM import MotionPerceiver, MotionPercieverWSignals
