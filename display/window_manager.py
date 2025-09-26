"""
WindowManager class for handling GLFW window creation and management.

This module provides window lifecycle management, framebuffer handling,
and event polling for the black hole simulation.
"""

import glfw
from typing import Tuple, Optional, Callable
from OpenGL.GL import *


class WindowManager:
    """Manages GLFW window creation, lifecycle, and basic event handling."""
    
    def __init__(self):
        """Initialize the WindowManager."""
        self._window: Optional[glfw._GLFWwindow] = None
        self._width: int = 0
        self._height: int = 0
        self._title: str = ""
        self._framebuffer_size_callback: Optional[Callable] = None
        
    def create_window(self, width: int, height: int, title: str = "Black Hole Simulation") -> bool:
        """
        Create and initialize a GLFW window with OpenGL context.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels  
            title: Window title string
            
        Returns:
            bool: True if window creation succeeded, False otherwise
        """
        # Initialize GLFW
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
            
        # Configure GLFW window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.RESIZABLE, True)
        
        # Create window
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False
            
        # Store window properties
        self._width = width
        self._height = height
        self._title = title
        
        # Make OpenGL context current
        glfw.make_context_current(self._window)
        
        # Set up framebuffer size callback
        glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_size_changed)
        
        # Enable vsync
        glfw.swap_interval(1)
        
        # Initialize OpenGL viewport
        glViewport(0, 0, width, height)
        
        return True
        
    def should_close(self) -> bool:
        """
        Check if the window should close.
        
        Returns:
            bool: True if window should close, False otherwise
        """
        if not self._window:
            return True
        return glfw.window_should_close(self._window)
        
    def swap_buffers(self) -> None:
        """Swap front and back buffers."""
        if self._window:
            glfw.swap_buffers(self._window)
            
    def poll_events(self) -> None:
        """Poll for and process events."""
        glfw.poll_events()
        
    def get_framebuffer_size(self) -> Tuple[int, int]:
        """
        Get the current framebuffer size.
        
        Returns:
            Tuple[int, int]: Width and height of framebuffer in pixels
        """
        if not self._window:
            return (0, 0)
        return glfw.get_framebuffer_size(self._window)
        
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get the current window size.
        
        Returns:
            Tuple[int, int]: Width and height of window in screen coordinates
        """
        if not self._window:
            return (0, 0)
        return glfw.get_window_size(self._window)
        
    def set_framebuffer_size_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Set callback for framebuffer size changes.
        
        Args:
            callback: Function to call when framebuffer size changes
        """
        self._framebuffer_size_callback = callback
        
    def get_window_handle(self) -> Optional[glfw._GLFWwindow]:
        """
        Get the GLFW window handle.
        
        Returns:
            Optional[glfw._GLFWwindow]: The GLFW window handle or None
        """
        return self._window
        
    def destroy(self) -> None:
        """Clean up and destroy the window."""
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()
        
    def _on_framebuffer_size_changed(self, window, width: int, height: int) -> None:
        """
        Internal callback for framebuffer size changes.
        
        Args:
            window: GLFW window handle
            width: New framebuffer width
            height: New framebuffer height
        """
        # Update OpenGL viewport
        glViewport(0, 0, width, height)
        
        # Call user callback if set
        if self._framebuffer_size_callback:
            self._framebuffer_size_callback(width, height)
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.destroy()