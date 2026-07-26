"""Public entry points for the AAI platform SDK."""

from importlib.metadata import PackageNotFoundError, version

from aai_core.context import PlatformContext, bootstrap
from aai_core.runtime import PlatformSettings

try:
    __version__ = version("aai-core")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.3.0"

__all__ = [
    "PlatformContext",
    "PlatformSettings",
    "__version__",
    "bootstrap",
]
