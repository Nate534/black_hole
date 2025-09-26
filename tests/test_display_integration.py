"""
Integration tests for display module components.

Tests the interaction between WindowManager and InputHandler.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from display import WindowManager, InputHandler, CameraState


class TestDisplayIntegration:
    """Integration tests for display module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.window_manager = WindowManager()
        self.input_handler = InputHandler()
    
    @patch('display.window_manager.glfw.init')
    @patch('display.window_manager.glfw.create_window')
    @patch('display.window_manager.glfw.make_context_current')
    @patch('display.window_manager.glfw.set_framebuffer_size_callback')
    @patch('display.window_manager.glfw.swap_interval')
    @patch('display.window_manager.glViewport')
    @patch('display.input_handler.InputHandler._setup_callbacks')
    def test_window_manager_input_handler_integration(self, mock_setup_callbacks, 
                                                    mock_viewport, mock_swap_interval,
                                                    mock_set_callback, mock_make_current,
                                                    mock_create_window, mock_init):
        """Test WindowManager and InputHandler working together."""
        # Arrange
        mock_init.return_value = True
        mock_window = MagicMock()
        mock_create_window.return_value = mock_window
        
        # Act - Create window and set up input handler
        window_created = self.window_manager.create_window(800, 600, "Test Integration")
        assert window_created is True
        
        window_handle = self.window_manager.get_window_handle()
        self.input_handler.set_window(window_handle)
        
        # Assert
        assert self.input_handler._window == window_handle
        mock_setup_callbacks.assert_called_once()
    
    def test_camera_state_management(self):
        """Test camera state management through InputHandler."""
        # Test initial state
        initial_state = self.input_handler.get_camera_state()
        assert isinstance(initial_state, CameraState)
        assert initial_state.position_z == 5.0
        
        # Test state modification
        new_state = CameraState(
            position_x=10.0,
            position_y=20.0,
            position_z=30.0,
            rotation_x=0.5,
            rotation_y=1.0,
            zoom=2.0
        )
        
        self.input_handler.set_camera_state(new_state)
        retrieved_state = self.input_handler.get_camera_state()
        
        assert retrieved_state.position_x == 10.0
        assert retrieved_state.position_y == 20.0
        assert retrieved_state.position_z == 30.0
        assert retrieved_state.rotation_x == 0.5
        assert retrieved_state.rotation_y == 1.0
        assert retrieved_state.zoom == 2.0
    
    def test_input_callback_registration(self):
        """Test input callback registration and management."""
        key_callback = Mock()
        mouse_callback = Mock()
        scroll_callback = Mock()
        
        # Register callbacks
        self.input_handler.register_key_callback(32, key_callback)  # Space key
        self.input_handler.register_mouse_button_callback(0, mouse_callback)  # Left mouse
        self.input_handler.register_scroll_callback(scroll_callback)
        
        # Verify callbacks are stored
        assert 32 in self.input_handler._key_callbacks
        assert 0 in self.input_handler._mouse_button_callbacks
        assert scroll_callback in self.input_handler._scroll_callbacks
    
    @patch('display.window_manager.glfw.destroy_window')
    @patch('display.window_manager.glfw.terminate')
    def test_cleanup_integration(self, mock_terminate, mock_destroy_window):
        """Test proper cleanup of resources."""
        # Set up mock window
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        
        # Test context manager cleanup
        with self.window_manager as wm:
            assert wm == self.window_manager
        
        # Verify cleanup was called
        mock_destroy_window.assert_called_once_with(mock_window)
        mock_terminate.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])