"""
Display module for black hole simulation.

This module handles window management, input processing, and UI controls.
"""

from .window_manager import WindowManager
from .input_handler import InputHandler, CameraState

__all__ = ['WindowManager', 'InputHandler', 'CameraState']