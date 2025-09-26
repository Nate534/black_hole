"""
GPU rendering module for black hole simulation.

This module manages OpenGL resources, shaders, and GPU compute operations.
"""

from .shader_manager import ShaderManager, ShaderCompilationError
from .buffer_manager import BufferManager, BufferAllocationError
from .compute_renderer import ComputeRenderer

__all__ = [
    'ShaderManager',
    'ShaderCompilationError',
    'BufferManager', 
    'BufferAllocationError',
    'ComputeRenderer'
]