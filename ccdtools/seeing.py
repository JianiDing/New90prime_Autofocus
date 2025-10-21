"""Compatibility shim for focus analysis.

Legacy code imported focus/seeing helpers from :mod:`ccdtools.seeing`.
The implementation now lives in :mod:`ccdtools.focus`.  This module simply
re-exports the public API so existing imports continue to work, but new
code should switch to :mod:`ccdtools.focus`.
"""

from .focus import (  # noqa: F401
    AmpAnalysis,
    FocusAmpAnalysis,
    FocusConfig,
    SeeingConfig,
    aggregate_results,
    analyze_amplifier,
    launch_gui,
)

__all__ = [
    "FocusConfig",
    "FocusAmpAnalysis",
    "SeeingConfig",
    "AmpAnalysis",
    "analyze_amplifier",
    "aggregate_results",
    "launch_gui",
]
