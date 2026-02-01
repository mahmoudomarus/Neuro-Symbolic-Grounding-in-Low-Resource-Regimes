"""
Tools: protocol and registry for the "Dictionary" layer (APIs, calculator, etc.).
"""
from .base import Tool
from .library import CalculatorTool, WikiTool

__all__ = ["Tool", "CalculatorTool", "WikiTool"]
