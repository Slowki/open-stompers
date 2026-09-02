"""Hardware model variants."""

from open_stompers_hardware_models.robots.microduck.variants import MicroduckVariant
from open_stompers_hardware_models.robots.mini_pi_plus.variants import MiniPiPlusVariant

type HardwareVariant = MicroduckVariant | MiniPiPlusVariant

__all__ = ["HardwareVariant", "MicroduckVariant", "MiniPiPlusVariant"]
