"""LoongFlow Agents - Pre-built agent implementations.

This package provides ready-to-use agent implementations for specific domains:
- Mathematical optimization and discovery
- Machine learning competitions
- General evolutionary algorithms
"""

from . import math_agent
from . import ml_agent
from . import general_agent

__all__ = ["math_agent", "ml_agent", "general_agent"]
