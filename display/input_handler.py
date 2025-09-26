"""
InputHandler class for processing keyboard and mouse input.

This module provides input processing for camera controls, user interaction,
and parameter adjustment in the black hole simulation.
"""

import glfw
from typing import Tuple, Dict, Callable, Optional, Set
from dataclasses import dataclass
import math


@dataclass
class CameraState:
    """Camera state for 3D navigation."""
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 5.0
    rotation_x: float = 0.0  # Pitch
    rotation_y: float = 0.0  # Yaw
    zoom: float = 1.0
    

class InputHandler:
    """Handles keyboard and mouse input for camera control and user interaction."""
    
    def __init__(self, window_handle: Optional[glfw._GLFWwindow] = None):
        """
        Initialize the InputHandler.
        
        Args:
            window_handle: GLFW window handle for input callbacks
        """
        self._window = window_handle
        self._camera_state = CameraState()
        
        # Input state tracking
        self._keys_pressed: Set[int] = set()
        self._mouse_buttons_pressed: Set[int] = set()
        self._last_mouse_x: float = 0.0
        self._last_mouse_y: float = 0.0
        self._mouse_sensitivity: float = 0.005
        self._zoom_sensitivity: float = 0.1
        self._movement_speed: float = 2.0
        
        # Input callbacks
        self._key_callbacks: Dict[int, Callable[[int, int], None]] = {}
        self._mouse_button_callbacks: Dict[int, Callable[[int, int], None]] = {}
        self._scroll_callbacks: list[Callable[[float, float], None]] = []
        
        # Camera control settings
        self._camera_enabled: bool = True
        self._invert_y: bool = False
        
        if self._window:
            self._setup_callbacks()
    
    def set_window(self, window_handle: glfw._GLFWwindow) -> None:
        """
        Set the GLFW window handle and setup callbacks.
        
        Args:
            window_handle: GLFW window handle
        """
        self._window = window_handle
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Setup GLFW input callbacks."""
        if not self._window:
            return
            
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_mouse_move)
        glfw.set_scroll_callback(self._window, self._on_scroll)
    
    def process_keyboard(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """
        Process keyboard input events.
        
        Args:
            window: GLFW window handle
            key: Key code
            scancode: Platform-specific scancode
            action: Key action (press, release, repeat)
            mods: Modifier keys
        """
        if action == glfw.PRESS:
            self._keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self._keys_pressed.discard(key)
        
        # Call registered callbacks
        if key in self._key_callbacks:
            self._key_callbacks[key](action, mods)
    
    def process_mouse(self, window, button: int, action: int, mods: int) -> None:
        """
        Process mouse button events.
        
        Args:
            window: GLFW window handle
            button: Mouse button code
            action: Button action (press, release)
            mods: Modifier keys
        """
        if action == glfw.PRESS:
            self._mouse_buttons_pressed.add(button)
        elif action == glfw.RELEASE:
            self._mouse_buttons_pressed.discard(button)
        
        # Call registered callbacks
        if button in self._mouse_button_callbacks:
            self._mouse_button_callbacks[button](action, mods)
    
    def get_mouse_position(self) -> Tuple[float, float]:
        """
        Get current mouse position.
        
        Returns:
            Tuple[float, float]: Mouse x, y coordinates
        """
        if not self._window:
            return (0.0, 0.0)
        return glfw.get_cursor_pos(self._window)
    
    def is_key_pressed(self, key: int) -> bool:
        """
        Check if a key is currently pressed.
        
        Args:
            key: Key code to check
            
        Returns:
            bool: True if key is pressed, False otherwise
        """
        return key in self._keys_pressed
    
    def is_mouse_button_pressed(self, button: int) -> bool:
        """
        Check if a mouse button is currently pressed.
        
        Args:
            button: Mouse button code to check
            
        Returns:
            bool: True if button is pressed, False otherwise
        """
        return button in self._mouse_buttons_pressed
    
    def update_camera_controls(self, dt: float) -> None:
        """
        Update camera state based on current input.
        
        Args:
            dt: Delta time in seconds
        """
        if not self._camera_enabled:
            return
        
        # Movement controls (WASD + QE for up/down)
        movement_delta = self._movement_speed * dt
        
        if self.is_key_pressed(glfw.KEY_W):
            self._camera_state.position_z -= movement_delta
        if self.is_key_pressed(glfw.KEY_S):
            self._camera_state.position_z += movement_delta
        if self.is_key_pressed(glfw.KEY_A):
            self._camera_state.position_x -= movement_delta
        if self.is_key_pressed(glfw.KEY_D):
            self._camera_state.position_x += movement_delta
        if self.is_key_pressed(glfw.KEY_Q):
            self._camera_state.position_y -= movement_delta
        if self.is_key_pressed(glfw.KEY_E):
            self._camera_state.position_y += movement_delta
        
        # Mouse look (when right mouse button is held)
        if self.is_mouse_button_pressed(glfw.MOUSE_BUTTON_RIGHT):
            mouse_x, mouse_y = self.get_mouse_position()
            
            if hasattr(self, '_mouse_initialized'):
                delta_x = mouse_x - self._last_mouse_x
                delta_y = mouse_y - self._last_mouse_y
                
                if self._invert_y:
                    delta_y = -delta_y
                
                self._camera_state.rotation_y += delta_x * self._mouse_sensitivity
                self._camera_state.rotation_x += delta_y * self._mouse_sensitivity
                
                # Clamp pitch to prevent gimbal lock
                self._camera_state.rotation_x = max(-math.pi/2 + 0.1, 
                                                  min(math.pi/2 - 0.1, self._camera_state.rotation_x))
            else:
                self._mouse_initialized = True
            
            self._last_mouse_x = mouse_x
            self._last_mouse_y = mouse_y
        else:
            # Reset mouse initialization when not looking
            if hasattr(self, '_mouse_initialized'):
                delattr(self, '_mouse_initialized')
    
    def get_camera_state(self) -> CameraState:
        """
        Get current camera state.
        
        Returns:
            CameraState: Current camera position, rotation, and zoom
        """
        return self._camera_state
    
    def set_camera_state(self, state: CameraState) -> None:
        """
        Set camera state.
        
        Args:
            state: New camera state
        """
        self._camera_state = state
    
    def reset_camera(self) -> None:
        """Reset camera to default position and orientation."""
        self._camera_state = CameraState()
    
    def set_camera_enabled(self, enabled: bool) -> None:
        """
        Enable or disable camera controls.
        
        Args:
            enabled: True to enable camera controls, False to disable
        """
        self._camera_enabled = enabled
    
    def set_mouse_sensitivity(self, sensitivity: float) -> None:
        """
        Set mouse sensitivity for camera rotation.
        
        Args:
            sensitivity: Mouse sensitivity multiplier
        """
        self._mouse_sensitivity = max(0.001, min(0.1, sensitivity))
    
    def set_movement_speed(self, speed: float) -> None:
        """
        Set camera movement speed.
        
        Args:
            speed: Movement speed in units per second
        """
        self._movement_speed = max(0.1, min(10.0, speed))
    
    def set_invert_y(self, invert: bool) -> None:
        """
        Set Y-axis inversion for mouse look.
        
        Args:
            invert: True to invert Y-axis, False for normal
        """
        self._invert_y = invert
    
    def register_key_callback(self, key: int, callback: Callable[[int, int], None]) -> None:
        """
        Register a callback for specific key events.
        
        Args:
            key: Key code to listen for
            callback: Function to call with (action, mods) parameters
        """
        self._key_callbacks[key] = callback
    
    def register_mouse_button_callback(self, button: int, callback: Callable[[int, int], None]) -> None:
        """
        Register a callback for specific mouse button events.
        
        Args:
            button: Mouse button code to listen for
            callback: Function to call with (action, mods) parameters
        """
        self._mouse_button_callbacks[button] = callback
    
    def register_scroll_callback(self, callback: Callable[[float, float], None]) -> None:
        """
        Register a callback for scroll events.
        
        Args:
            callback: Function to call with (x_offset, y_offset) parameters
        """
        self._scroll_callbacks.append(callback)
    
    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """Internal GLFW key callback."""
        self.process_keyboard(window, key, scancode, action, mods)
    
    def _on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
        """Internal GLFW mouse button callback."""
        self.process_mouse(window, button, action, mods)
    
    def _on_mouse_move(self, window, x_pos: float, y_pos: float) -> None:
        """Internal GLFW mouse movement callback."""
        # Mouse movement is handled in update_camera_controls
        pass
    
    def _on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        """Internal GLFW scroll callback."""
        # Handle zoom
        if self._camera_enabled:
            zoom_delta = y_offset * self._zoom_sensitivity
            self._camera_state.zoom = max(0.1, min(10.0, self._camera_state.zoom + zoom_delta))
        
        # Call registered scroll callbacks
        for callback in self._scroll_callbacks:
            callback(x_offset, y_offset)