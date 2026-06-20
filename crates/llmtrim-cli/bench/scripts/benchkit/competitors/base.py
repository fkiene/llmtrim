"""The Competitor interface the benchkit engine drives.

A competitor is the OTHER library in an llmtrim-vs-X comparison. The engine (sweep + live +
report) never names a specific tool; it iterates `config_grid()`, calls `compress()`, asks
`ml_fired()`, and pulls report caveats from `notes()`. To add a tool, implement this
interface and register it (see competitors/__init__.py); the numbers and layout are shared.
"""
from abc import ABC, abstractmethod


class Competitor(ABC):
    name: str  # registry key, used in labels + report text (e.g. "headroom")
    display: str  # human label (e.g. "Headroom")

    @abstractmethod
    def compress(self, messages, cfg, repeats):
        """Compress `messages` under `cfg` (one entry from config_grid()), timing `repeats`
        runs. Returns (out_messages, transforms, median_ms)."""

    @abstractmethod
    def config_grid(self):
        """A list of (label, cfg_dict) spanning no-op .. max aggressiveness. Labels are the
        arm names used throughout the report, so keep them stable + generic."""

    @abstractmethod
    def ml_fired(self, transforms):
        """Whether the ML path fired for a given compress() call (from its transforms)."""

    def disable_ml(self):
        """Disable the competitor's ML path for the whole process. Called once at startup when
        --no-ml is set. No-op when the competitor has no ML path."""

    @abstractmethod
    def notes(self):
        """A dict of report caveat strings specific to this competitor. The engine/report
        stays generic and stitches these in. Keys are stable so report.render() can place
        them; missing keys are simply skipped."""
