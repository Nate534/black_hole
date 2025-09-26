"""
Unit tests for InputHandler class.

Tests input processing, camera controls, and event handling.
"""

import pytest
import glfw
import math
from unittest.mock import Mock, patch, MagicMock
from display.input_handler import InputHandler, CameraState


class TestInputHandler:
    """Test cases for InputHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_handler = InputHandler()
        
    def test_init_without_window(self):
        """Test initialization without window handle."""
        handler = InputHandler()
        assert handler._window is None
        assert isinstance(handler._camera_state, CameraState)
        assert handler._camera_enabled is True
        
    @patch('glfw.set_key_callback')
    @patch('glfw.set_mouse_button_callback')
    @patch('glfw.set_cursor_pos_callback')
    @patch('glfw.set_scroll_callback')
    def test_init_with_window(self, mock_scroll, mock_cursor, mock_mouse, mock_key):
        """Test initialization with window handle."""
        mock_window = MagicMock()
        handler = InputHandler(mock_window)
        
        assert handler._window == mock_window
        mock_key.assert_called_once()
        mock_mouse.assert_called_once()
        mock_cursor.assert_called_once()
        mock_scroll.assert_called_once()
    
    @patch('glfw.set_key_callback')
    @patch('glfw.set_mouse_button_callback')
    @patch('glfw.set_cursor_pos_callback')
    @patch('glfw.set_scroll_callback')
    def test_set_window(self, mock_scroll, mock_cursor, mock_mouse, mock_key):
        """Test setting window handle."""
        mock_window = MagicMock()
        self.input_handler.set_window(mock_window)
        
        assert self.input_handler._window == mock_window
        mock_key.assert_called_once()
        mock_mouse.assert_called_once()
        mock_cursor.assert_called_once()
        mock_scroll.assert_called_once()
    
    def test_process_keyboard_press(self):
        """Test keyboard press processing."""
        mock_window = MagicMock()
        key = glfw.KEY_W
        
        self.input_handler.process_keyboard(mock_window, key, 0, glfw.PRESS, 0)
        
        assert key in self.input_handler._keys_pressed
        assert self.input_handler.is_key_pressed(key)
    
    def test_process_keyboard_release(self):
        """Test keyboard release processing."""
        mock_window = MagicMock()
        key = glfw.KEY_W
        
        # First press the key
        self.input_handler.process_keyboard(mock_window, key, 0, glfw.PRESS, 0)
        assert self.input_handler.is_key_pressed(key)
        
        # Then release it
        self.input_handler.process_keyboard(mock_window, key, 0, glfw.RELEASE, 0)
        assert not self.input_handler.is_key_pressed(key)
    
    def test_process_keyboard_callback(self):
        """Test keyboard callback registration and execution."""
        mock_window = MagicMock()
        key = glfw.KEY_SPACE
        callback = Mock()
        
        self.input_handler.register_key_callback(key, callback)
        self.input_handler.process_keyboard(mock_window, key, 0, glfw.PRESS, 0)
        
        callback.assert_called_once_with(glfw.PRESS, 0)
    
    def test_process_mouse_press(self):
        """Test mouse button press processing."""
        mock_window = MagicMock()
        button = glfw.MOUSE_BUTTON_LEFT
        
        self.input_handler.process_mouse(mock_window, button, glfw.PRESS, 0)
        
        assert button in self.input_handler._mouse_buttons_pressed
        assert self.input_handler.is_mouse_button_pressed(button)
    
    def test_process_mouse_release(self):
        """Test mouse button release processing."""
        mock_window = MagicMock()
        button = glfw.MOUSE_BUTTON_LEFT
        
        # First press the button
        self.input_handler.process_mouse(mock_window, button, glfw.PRESS, 0)
        assert self.input_handler.is_mouse_button_pressed(button)
        
        # Then release it
        self.input_handler.process_mouse(mock_window, button, glfw.RELEASE, 0)
        assert not self.input_handler.is_mouse_button_pressed(button)
    
    def test_process_mouse_callback(self):
        """Test mouse button callback registration and execution."""
        mock_window = MagicMock()
        button = glfw.MOUSE_BUTTON_RIGHT
        callback = Mock()
        
        self.input_handler.register_mouse_button_callback(button, callback)
        self.input_handler.process_mouse(mock_window, button, glfw.PRESS, 0)
        
        callback.assert_called_once_with(glfw.PRESS, 0)
    
    @patch('glfw.get_cursor_pos')
    @patch('display.input_handler.InputHandler._setup_callbacks')
    def test_get_mouse_position(self, mock_setup_callbacks, mock_get_cursor_pos):
        """Test getting mouse position."""
        mock_window = MagicMock()
        self.input_handler.set_window(mock_window)
        mock_get_cursor_pos.return_value = (100.0, 200.0)
        
        x, y = self.input_handler.get_mouse_position()
        
        assert x == 100.0
        assert y == 200.0
        mock_get_cursor_pos.assert_called_once_with(mock_window)
    
    def test_get_mouse_position_no_window(self):
        """Test getting mouse position without window."""
        x, y = self.input_handler.get_mouse_position()
        assert x == 0.0
        assert y == 0.0
    
    def test_camera_movement_controls(self):
        """Test camera movement with WASD keys."""
        mock_window = MagicMock()
        dt = 0.016  # ~60 FPS
        
        # Simulate pressing W key (forward)
        self.input_handler.process_keyboard(mock_window, glfw.KEY_W, 0, glfw.PRESS, 0)
        initial_z = self.input_handler._camera_state.position_z
        
        self.input_handler.update_camera_controls(dt)
        
        # Camera should move forward (negative Z)
        assert self.input_handler._camera_state.position_z < initial_z
    
    def test_camera_movement_all_directions(self):
        """Test camera movement in all directions."""
        mock_window = MagicMock()
        dt = 0.016
        
        # Test all movement keys
        keys_and_expected_changes = [
            (glfw.KEY_W, 'position_z', lambda old, new: new < old),  # Forward
            (glfw.KEY_S, 'position_z', lambda old, new: new > old),  # Backward
            (glfw.KEY_A, 'position_x', lambda old, new: new < old),  # Left
            (glfw.KEY_D, 'position_x', lambda old, new: new > old),  # Right
            (glfw.KEY_Q, 'position_y', lambda old, new: new < old),  # Down
            (glfw.KEY_E, 'position_y', lambda old, new: new > old),  # Up
        ]
        
        for key, attr, comparison in keys_and_expected_changes:
            # Reset camera and clear pressed keys
            self.input_handler.reset_camera()
            self.input_handler._keys_pressed.clear()
            initial_value = getattr(self.input_handler._camera_state, attr)
            
            # Press key and update (key stays pressed during update)
            self.input_handler.process_keyboard(mock_window, key, 0, glfw.PRESS, 0)
            self.input_handler.update_camera_controls(dt)
            
            new_value = getattr(self.input_handler._camera_state, attr)
            assert comparison(initial_value, new_value), f"Key {key} didn't move {attr} correctly"
            
            # Release key for next test
            self.input_handler.process_keyboard(mock_window, key, 0, glfw.RELEASE, 0)
    
    @patch('glfw.get_cursor_pos')
    @patch('display.input_handler.InputHandler._setup_callbacks')
    def test_mouse_look_controls(self, mock_setup_callbacks, mock_get_cursor_pos):
        """Test mouse look camera controls."""
        mock_window = MagicMock()
        self.input_handler.set_window(mock_window)
        
        # Simulate right mouse button press
        self.input_handler.process_mouse(mock_window, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
        
        # First update - initialize mouse position
        mock_get_cursor_pos.return_value = (400.0, 300.0)
        self.input_handler.update_camera_controls(0.016)
        
        # Second update - mouse movement
        mock_get_cursor_pos.return_value = (450.0, 250.0)  # Move right and up
        initial_rotation_y = self.input_handler._camera_state.rotation_y
        initial_rotation_x = self.input_handler._camera_state.rotation_x
        
        self.input_handler.update_camera_controls(0.016)
        
        # Camera should rotate
        assert self.input_handler._camera_state.rotation_y != initial_rotation_y
        assert self.input_handler._camera_state.rotation_x != initial_rotation_x
    
    def test_camera_pitch_clamping(self):
        """Test that camera pitch is clamped to prevent gimbal lock."""
        # Set extreme rotation values
        self.input_handler._camera_state.rotation_x = math.pi  # 180 degrees
        
        mock_window = MagicMock()
        self.input_handler.process_mouse(mock_window, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
        
        with patch('glfw.get_cursor_pos') as mock_get_cursor_pos:
            # Initialize mouse
            mock_get_cursor_pos.return_value = (400.0, 300.0)
            self.input_handler.update_camera_controls(0.016)
            
            # Try to rotate beyond limit
            mock_get_cursor_pos.return_value = (400.0, 100.0)  # Large upward movement
            self.input_handler.update_camera_controls(0.016)
        
        # Pitch should be clamped
        assert self.input_handler._camera_state.rotation_x < math.pi/2
        assert self.input_handler._camera_state.rotation_x > -math.pi/2
    
    def test_scroll_zoom_control(self):
        """Test scroll wheel zoom control."""
        mock_window = MagicMock()
        initial_zoom = self.input_handler._camera_state.zoom
        
        # Simulate scroll up (zoom in)
        self.input_handler._on_scroll(mock_window, 0.0, 1.0)
        
        assert self.input_handler._camera_state.zoom > initial_zoom
    
    def test_scroll_callback_registration(self):
        """Test scroll callback registration and execution."""
        callback = Mock()
        self.input_handler.register_scroll_callback(callback)
        
        mock_window = MagicMock()
        self.input_handler._on_scroll(mock_window, 1.0, 2.0)
        
        callback.assert_called_once_with(1.0, 2.0)
    
    def test_camera_state_get_set(self):
        """Test getting and setting camera state."""
        new_state = CameraState(
            position_x=1.0,
            position_y=2.0,
            position_z=3.0,
            rotation_x=0.5,
            rotation_y=1.0,
            zoom=2.0
        )
        
        self.input_handler.set_camera_state(new_state)
        retrieved_state = self.input_handler.get_camera_state()
        
        assert retrieved_state.position_x == 1.0
        assert retrieved_state.position_y == 2.0
        assert retrieved_state.position_z == 3.0
        assert retrieved_state.rotation_x == 0.5
        assert retrieved_state.rotation_y == 1.0
        assert retrieved_state.zoom == 2.0
    
    def test_reset_camera(self):
        """Test camera reset functionality."""
        # Modify camera state
        self.input_handler._camera_state.position_x = 10.0
        self.input_handler._camera_state.rotation_y = 1.5
        
        # Reset camera
        self.input_handler.reset_camera()
        
        # Should be back to defaults
        assert self.input_handler._camera_state.position_x == 0.0
        assert self.input_handler._camera_state.position_y == 0.0
        assert self.input_handler._camera_state.position_z == 5.0
        assert self.input_handler._camera_state.rotation_x == 0.0
        assert self.input_handler._camera_state.rotation_y == 0.0
        assert self.input_handler._camera_state.zoom == 1.0
    
    def test_camera_enable_disable(self):
        """Test enabling and disabling camera controls."""
        mock_window = MagicMock()
        dt = 0.016
        
        # Disable camera
        self.input_handler.set_camera_enabled(False)
        
        # Try to move camera
        self.input_handler.process_keyboard(mock_window, glfw.KEY_W, 0, glfw.PRESS, 0)
        initial_z = self.input_handler._camera_state.position_z
        
        self.input_handler.update_camera_controls(dt)
        
        # Camera should not move
        assert self.input_handler._camera_state.position_z == initial_z
        
        # Re-enable camera
        self.input_handler.set_camera_enabled(True)
        self.input_handler.update_camera_controls(dt)
        
        # Now camera should move
        assert self.input_handler._camera_state.position_z < initial_z
    
    def test_mouse_sensitivity_setting(self):
        """Test mouse sensitivity setting with bounds."""
        # Test normal value
        self.input_handler.set_mouse_sensitivity(0.01)
        assert self.input_handler._mouse_sensitivity == 0.01
        
        # Test lower bound
        self.input_handler.set_mouse_sensitivity(0.0001)
        assert self.input_handler._mouse_sensitivity == 0.001
        
        # Test upper bound
        self.input_handler.set_mouse_sensitivity(0.5)
        assert self.input_handler._mouse_sensitivity == 0.1
    
    def test_movement_speed_setting(self):
        """Test movement speed setting with bounds."""
        # Test normal value
        self.input_handler.set_movement_speed(5.0)
        assert self.input_handler._movement_speed == 5.0
        
        # Test lower bound
        self.input_handler.set_movement_speed(0.05)
        assert self.input_handler._movement_speed == 0.1
        
        # Test upper bound
        self.input_handler.set_movement_speed(20.0)
        assert self.input_handler._movement_speed == 10.0
    
    def test_invert_y_setting(self):
        """Test Y-axis inversion setting."""
        self.input_handler.set_invert_y(True)
        assert self.input_handler._invert_y is True
        
        self.input_handler.set_invert_y(False)
        assert self.input_handler._invert_y is False


if __name__ == '__main__':
    pytest.main([__file__])