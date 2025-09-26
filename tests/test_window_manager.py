"""
Unit tests for WindowManager class.

Tests window creation, lifecycle management, and framebuffer handling.
"""

import pytest
import glfw
from unittest.mock import Mock, patch, MagicMock
from display.window_manager import WindowManager


class TestWindowManager:
    """Test cases for WindowManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.window_manager = WindowManager()
        
    def teardown_method(self):
        """Clean up after tests."""
        # Only destroy if we have a real window, not a mock
        if (hasattr(self.window_manager, '_window') and 
            self.window_manager._window and 
            not isinstance(self.window_manager._window, MagicMock)):
            self.window_manager.destroy()
        # Reset window to None for mocked tests
        self.window_manager._window = None
    
    @patch('glfw.init')
    @patch('glfw.create_window')
    @patch('glfw.make_context_current')
    @patch('glfw.set_framebuffer_size_callback')
    @patch('glfw.swap_interval')
    @patch('display.window_manager.glViewport')
    def test_create_window_success(self, mock_viewport, mock_swap_interval, 
                                 mock_set_callback, mock_make_current, 
                                 mock_create_window, mock_init):
        """Test successful window creation."""
        # Arrange
        mock_init.return_value = True
        mock_window = MagicMock()
        mock_create_window.return_value = mock_window
        
        # Act
        result = self.window_manager.create_window(800, 600, "Test Window")
        
        # Assert
        assert result is True
        assert self.window_manager._window == mock_window
        assert self.window_manager._width == 800
        assert self.window_manager._height == 600
        assert self.window_manager._title == "Test Window"
        
        mock_init.assert_called_once()
        mock_create_window.assert_called_once()
        mock_make_current.assert_called_once_with(mock_window)
        mock_set_callback.assert_called_once()
        mock_swap_interval.assert_called_once_with(1)
        mock_viewport.assert_called_once_with(0, 0, 800, 600)
    
    @patch('glfw.init')
    def test_create_window_glfw_init_failure(self, mock_init):
        """Test window creation when GLFW init fails."""
        # Arrange
        mock_init.return_value = False
        
        # Act
        result = self.window_manager.create_window(800, 600)
        
        # Assert
        assert result is False
        assert self.window_manager._window is None
    
    @patch('glfw.init')
    @patch('glfw.create_window')
    @patch('glfw.terminate')
    def test_create_window_creation_failure(self, mock_terminate, mock_create_window, mock_init):
        """Test window creation when window creation fails."""
        # Arrange
        mock_init.return_value = True
        mock_create_window.return_value = None
        
        # Act
        result = self.window_manager.create_window(800, 600)
        
        # Assert
        assert result is False
        assert self.window_manager._window is None
        mock_terminate.assert_called_once()
    
    def test_should_close_no_window(self):
        """Test should_close when no window exists."""
        # Act & Assert
        assert self.window_manager.should_close() is True
    
    @patch('glfw.window_should_close')
    def test_should_close_with_window(self, mock_should_close):
        """Test should_close with existing window."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        mock_should_close.return_value = False
        
        # Act
        result = self.window_manager.should_close()
        
        # Assert
        assert result is False
        mock_should_close.assert_called_once_with(mock_window)
    
    @patch('glfw.swap_buffers')
    def test_swap_buffers(self, mock_swap_buffers):
        """Test buffer swapping."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        
        # Act
        self.window_manager.swap_buffers()
        
        # Assert
        mock_swap_buffers.assert_called_once_with(mock_window)
    
    def test_swap_buffers_no_window(self):
        """Test buffer swapping with no window."""
        # Act (should not raise exception)
        self.window_manager.swap_buffers()
    
    @patch('glfw.poll_events')
    def test_poll_events(self, mock_poll_events):
        """Test event polling."""
        # Act
        self.window_manager.poll_events()
        
        # Assert
        mock_poll_events.assert_called_once()
    
    @patch('glfw.get_framebuffer_size')
    def test_get_framebuffer_size(self, mock_get_size):
        """Test getting framebuffer size."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        mock_get_size.return_value = (1024, 768)
        
        # Act
        width, height = self.window_manager.get_framebuffer_size()
        
        # Assert
        assert width == 1024
        assert height == 768
        mock_get_size.assert_called_once_with(mock_window)
    
    def test_get_framebuffer_size_no_window(self):
        """Test getting framebuffer size with no window."""
        # Act
        width, height = self.window_manager.get_framebuffer_size()
        
        # Assert
        assert width == 0
        assert height == 0
    
    @patch('glfw.get_window_size')
    def test_get_window_size(self, mock_get_size):
        """Test getting window size."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        mock_get_size.return_value = (800, 600)
        
        # Act
        width, height = self.window_manager.get_window_size()
        
        # Assert
        assert width == 800
        assert height == 600
        mock_get_size.assert_called_once_with(mock_window)
    
    def test_set_framebuffer_size_callback(self):
        """Test setting framebuffer size callback."""
        # Arrange
        callback = Mock()
        
        # Act
        self.window_manager.set_framebuffer_size_callback(callback)
        
        # Assert
        assert self.window_manager._framebuffer_size_callback == callback
    
    def test_get_window_handle(self):
        """Test getting window handle."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        
        # Act
        handle = self.window_manager.get_window_handle()
        
        # Assert
        assert handle == mock_window
    
    @patch('glfw.destroy_window')
    @patch('glfw.terminate')
    def test_destroy(self, mock_terminate, mock_destroy_window):
        """Test window destruction."""
        # Arrange
        mock_window = MagicMock()
        self.window_manager._window = mock_window
        
        # Act
        self.window_manager.destroy()
        
        # Assert
        mock_destroy_window.assert_called_once_with(mock_window)
        mock_terminate.assert_called_once()
        assert self.window_manager._window is None
    
    @patch('display.window_manager.glViewport')
    def test_framebuffer_size_callback(self, mock_viewport):
        """Test internal framebuffer size callback."""
        # Arrange
        user_callback = Mock()
        self.window_manager.set_framebuffer_size_callback(user_callback)
        mock_window = MagicMock()
        
        # Act
        self.window_manager._on_framebuffer_size_changed(mock_window, 1920, 1080)
        
        # Assert
        mock_viewport.assert_called_once_with(0, 0, 1920, 1080)
        user_callback.assert_called_once_with(1920, 1080)
    
    @patch('display.window_manager.glViewport')
    def test_framebuffer_size_callback_no_user_callback(self, mock_viewport):
        """Test internal framebuffer size callback without user callback."""
        # Arrange
        mock_window = MagicMock()
        
        # Act
        self.window_manager._on_framebuffer_size_changed(mock_window, 1920, 1080)
        
        # Assert
        mock_viewport.assert_called_once_with(0, 0, 1920, 1080)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Arrange
        mock_window = MagicMock()
        
        # Act & Assert
        with patch.object(self.window_manager, 'destroy') as mock_destroy:
            with self.window_manager as wm:
                assert wm == self.window_manager
            mock_destroy.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])