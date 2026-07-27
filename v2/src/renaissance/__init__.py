"""Renaissance 2.0 — composable multimodal transformer platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("renaissance")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0+uninstalled"

from renaissance import components as components  # noqa: E402  (built-ins self-register on import)
from renaissance.model import RenaissanceModel
from renaissance.model_config import EncoderSpec, FusionSpec, ModelConfig

__all__ = [
    "EncoderSpec",
    "FusionSpec",
    "ModelConfig",
    "RenaissanceModel",
    "__version__",
]
