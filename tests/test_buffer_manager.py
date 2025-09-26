"""
Tests for BufferManager class.

Tests buffer creation, updates, and vertex array object management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gpu_rendering.buffer_manager import BufferManager, BufferAllocationError
from physics.particle import Particle
from physics.vector3 import Vector3


class TestBufferManager:
    """Test cases for BufferManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer_manager = BufferManager()
        
        # Create test particles
        self.test_particles = [
            Particle(
                mass=1.0,
                position=Vector3(1.0, 2.0, 3.0),
                velocity=Vector3(0.1, 0.2, 0.3)
            ),
            Particle(
                mass=2.0,
                position=Vector3(4.0, 5.0, 6.0),
                velocity=Vector3(0.4, 0.5, 0.6)
            ),
            Particle(
                mass=0.5,
                position=Vector3(7.0, 8.0, 9.0),
                velocity=Vector3(0.7, 0.8, 0.9)
            )
        ]
    
    def test_particles_to_array_conversion(self):
        """Test conversion of particles to numpy array."""
        particle_array = self.buffer_manager._particles_to_array(self.test_particles)
        
        # Check array shape
        assert particle_array.shape == (3, 8)
        assert particle_array.dtype == np.float32
        
        # Check first particle data
        np.testing.assert_array_almost_equal(
            particle_array[0], 
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 1.0]
        )
        
        # Check second particle data
        np.testing.assert_array_almost_equal(
            particle_array[1],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 2.0, 1.0]
        )
        
        # Check third particle data
        np.testing.assert_array_almost_equal(
            particle_array[2],
            [7.0, 8.0, 9.0, 0.7, 0.8, 0.9, 0.5, 1.0]
        )
    
    def test_particles_to_array_empty_list(self):
        """Test conversion of empty particle list."""
        particle_array = self.buffer_manager._particles_to_array([])
        
        assert particle_array.shape == (0, 8)
        assert particle_array.dtype == np.float32
    
    def test_particles_to_array_inactive_particle(self):
        """Test conversion with inactive particle."""
        # Create inactive particle
        inactive_particle = Particle(
            mass=1.0,
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            active=False  # Set inactive directly
        )
        
        particle_array = self.buffer_manager._particles_to_array([inactive_particle])
        
        # Check that active flag is 0.0
        assert particle_array[0, 7] == 0.0
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_particle_buffer_success(self, mock_gl):
        """Test successful particle buffer creation."""
        # Setup mocks
        mock_gl.glGenBuffers.return_value = 42
        mock_gl.glGetError.return_value = mock_gl.GL_NO_ERROR
        
        # Create buffer
        buffer_id = self.buffer_manager.create_particle_buffer(self.test_particles, "test_buffer")
        
        # Verify calls
        mock_gl.glGenBuffers.assert_called_with(1)
        # Check that buffer was bound with correct ID (not the final unbind call)
        bind_calls = [call for call in mock_gl.glBindBuffer.call_args_list 
                     if call[0][1] == 42]
        assert len(bind_calls) > 0, "Buffer should have been bound with correct ID"
        mock_gl.glBufferData.assert_called_once()
        
        # Check buffer data call
        args = mock_gl.glBufferData.call_args[0]
        assert args[0] == mock_gl.GL_SHADER_STORAGE_BUFFER
        assert args[3] == mock_gl.GL_DYNAMIC_DRAW
        
        # Verify return value and storage
        assert buffer_id == 42
        assert self.buffer_manager.get_buffer_id("test_buffer") == 42
        assert self.buffer_manager.get_buffer_size("test_buffer") == 3
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_particle_buffer_empty_particles(self, mock_gl):
        """Test particle buffer creation with empty particle list."""
        mock_gl.glGenBuffers.return_value = 42
        mock_gl.glGetError.return_value = mock_gl.GL_NO_ERROR
        
        # Create buffer with empty list
        buffer_id = self.buffer_manager.create_particle_buffer([], "empty_buffer")
        
        # Verify buffer created with default size
        assert buffer_id == 42
        assert self.buffer_manager.get_buffer_size("empty_buffer") == 1000
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_particle_buffer_generation_failure(self, mock_gl):
        """Test buffer creation failure during generation."""
        mock_gl.glGenBuffers.return_value = 0  # Failure
        
        with pytest.raises(BufferAllocationError, match="Failed to generate OpenGL buffer"):
            self.buffer_manager.create_particle_buffer(self.test_particles)
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_particle_buffer_opengl_error(self, mock_gl):
        """Test buffer creation failure due to OpenGL error."""
        mock_gl.glGenBuffers.return_value = 42
        mock_gl.glGetError.return_value = 1234  # Some OpenGL error
        
        with pytest.raises(BufferAllocationError, match="OpenGL error during buffer creation: 1234"):
            self.buffer_manager.create_particle_buffer(self.test_particles)
        
        # Verify cleanup
        mock_gl.glDeleteBuffers.assert_called_with(1, [42])
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_update_particle_buffer_success(self, mock_gl):
        """Test successful particle buffer update."""
        # Setup existing buffer
        self.buffer_manager._buffers["test"] = 42
        self.buffer_manager._buffer_sizes["test"] = 10
        mock_gl.glGetError.return_value = mock_gl.GL_NO_ERROR
        
        # Update buffer
        self.buffer_manager.update_particle_buffer("test", self.test_particles)
        
        # Verify calls
        # Check that buffer was bound with correct ID (not the final unbind call)
        bind_calls = [call for call in mock_gl.glBindBuffer.call_args_list 
                     if call[0][1] == 42]
        assert len(bind_calls) > 0, "Buffer should have been bound with correct ID"
        mock_gl.glBufferSubData.assert_called_once()
        
        # Check that buffer wasn't reallocated (size 3 < 10)
        mock_gl.glBufferData.assert_not_called()
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_update_particle_buffer_resize(self, mock_gl):
        """Test particle buffer update with resize."""
        # Setup existing buffer with small size
        self.buffer_manager._buffers["test"] = 42
        self.buffer_manager._buffer_sizes["test"] = 2  # Smaller than particle count
        mock_gl.glGetError.return_value = mock_gl.GL_NO_ERROR
        
        # Update buffer
        self.buffer_manager.update_particle_buffer("test", self.test_particles)
        
        # Verify buffer was reallocated
        mock_gl.glBufferData.assert_called_once()
        mock_gl.glBufferSubData.assert_not_called()
        
        # Check new size (should be max(3, 2*1.5) = 3)
        assert self.buffer_manager.get_buffer_size("test") == 3
    
    def test_update_particle_buffer_not_found(self):
        """Test error when updating non-existent buffer."""
        with pytest.raises(KeyError, match="Buffer 'nonexistent' not found"):
            self.buffer_manager.update_particle_buffer("nonexistent", self.test_particles)
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_update_particle_buffer_opengl_error(self, mock_gl):
        """Test buffer update failure due to OpenGL error."""
        self.buffer_manager._buffers["test"] = 42
        self.buffer_manager._buffer_sizes["test"] = 10
        mock_gl.glGetError.return_value = 1234  # OpenGL error
        
        with pytest.raises(BufferAllocationError, match="OpenGL error during buffer update: 1234"):
            self.buffer_manager.update_particle_buffer("test", self.test_particles)
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_vertex_array_success(self, mock_gl):
        """Test successful VAO creation."""
        mock_gl.glGenVertexArrays.return_value = 123
        
        vao_id = self.buffer_manager.create_vertex_array("test_vao")
        
        mock_gl.glGenVertexArrays.assert_called_with(1)
        assert vao_id == 123
        assert self.buffer_manager.get_vao_id("test_vao") == 123
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_create_vertex_array_failure(self, mock_gl):
        """Test VAO creation failure."""
        mock_gl.glGenVertexArrays.return_value = 0  # Failure
        
        with pytest.raises(BufferAllocationError, match="Failed to generate OpenGL vertex array"):
            self.buffer_manager.create_vertex_array()
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_setup_particle_vertex_attributes(self, mock_gl):
        """Test setting up particle vertex attributes."""
        # Setup VAO and buffer
        self.buffer_manager._vaos["test_vao"] = 123
        self.buffer_manager._buffers["test_buffer"] = 456
        
        # Setup vertex attributes
        self.buffer_manager.setup_particle_vertex_attributes("test_vao", "test_buffer")
        
        # Verify calls
        mock_gl.glBindVertexArray.assert_any_call(123)
        # Check that buffer was bound with correct ID (not the final unbind call)
        bind_calls = [call for call in mock_gl.glBindBuffer.call_args_list 
                     if call[0][1] == 456]
        assert len(bind_calls) > 0, "Buffer should have been bound with correct ID"
        
        # Verify vertex attribute setup (4 attributes)
        assert mock_gl.glVertexAttribPointer.call_count == 4
        assert mock_gl.glEnableVertexAttribArray.call_count == 4
        
        # Check specific attribute calls
        calls = mock_gl.glVertexAttribPointer.call_args_list
        
        # Position attribute (location 0)
        assert calls[0][0] == (0, 3, mock_gl.GL_FLOAT, mock_gl.GL_FALSE, 32, mock_gl.GLvoidp(0))
        
        # Velocity attribute (location 1)
        assert calls[1][0] == (1, 3, mock_gl.GL_FLOAT, mock_gl.GL_FALSE, 32, mock_gl.GLvoidp(12))
        
        # Mass attribute (location 2)
        assert calls[2][0] == (2, 1, mock_gl.GL_FLOAT, mock_gl.GL_FALSE, 32, mock_gl.GLvoidp(24))
        
        # Active attribute (location 3)
        assert calls[3][0] == (3, 1, mock_gl.GL_FLOAT, mock_gl.GL_FALSE, 32, mock_gl.GLvoidp(28))
    
    def test_setup_particle_vertex_attributes_vao_not_found(self):
        """Test error when VAO not found."""
        self.buffer_manager._buffers["test_buffer"] = 456
        
        with pytest.raises(KeyError, match="VAO 'missing_vao' not found"):
            self.buffer_manager.setup_particle_vertex_attributes("missing_vao", "test_buffer")
    
    def test_setup_particle_vertex_attributes_buffer_not_found(self):
        """Test error when buffer not found."""
        self.buffer_manager._vaos["test_vao"] = 123
        
        with pytest.raises(KeyError, match="Buffer 'missing_buffer' not found"):
            self.buffer_manager.setup_particle_vertex_attributes("test_vao", "missing_buffer")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_bind_buffer(self, mock_gl):
        """Test buffer binding to shader storage buffer binding point."""
        self.buffer_manager._buffers["test"] = 42
        
        self.buffer_manager.bind_buffer("test", 5)
        
        mock_gl.glBindBufferBase.assert_called_with(mock_gl.GL_SHADER_STORAGE_BUFFER, 5, 42)
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_bind_buffer_default_binding_point(self, mock_gl):
        """Test buffer binding with default binding point."""
        self.buffer_manager._buffers["test"] = 42
        
        self.buffer_manager.bind_buffer("test")
        
        mock_gl.glBindBufferBase.assert_called_with(mock_gl.GL_SHADER_STORAGE_BUFFER, 0, 42)
    
    def test_bind_buffer_not_found(self):
        """Test error when binding non-existent buffer."""
        with pytest.raises(KeyError, match="Buffer 'missing' not found"):
            self.buffer_manager.bind_buffer("missing")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_bind_vertex_array(self, mock_gl):
        """Test VAO binding."""
        self.buffer_manager._vaos["test"] = 123
        
        self.buffer_manager.bind_vertex_array("test")
        
        mock_gl.glBindVertexArray.assert_called_with(123)
    
    def test_bind_vertex_array_not_found(self):
        """Test error when binding non-existent VAO."""
        with pytest.raises(KeyError, match="VAO 'missing' not found"):
            self.buffer_manager.bind_vertex_array("missing")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_unbind_vertex_array(self, mock_gl):
        """Test VAO unbinding."""
        self.buffer_manager.unbind_vertex_array()
        
        mock_gl.glBindVertexArray.assert_called_with(0)
    
    def test_get_buffer_id(self):
        """Test getting buffer ID by name."""
        self.buffer_manager._buffers["test"] = 42
        assert self.buffer_manager.get_buffer_id("test") == 42
        
        with pytest.raises(KeyError, match="Buffer 'missing' not found"):
            self.buffer_manager.get_buffer_id("missing")
    
    def test_get_vao_id(self):
        """Test getting VAO ID by name."""
        self.buffer_manager._vaos["test"] = 123
        assert self.buffer_manager.get_vao_id("test") == 123
        
        with pytest.raises(KeyError, match="VAO 'missing' not found"):
            self.buffer_manager.get_vao_id("missing")
    
    def test_get_buffer_size(self):
        """Test getting buffer size."""
        self.buffer_manager._buffer_sizes["test"] = 100
        assert self.buffer_manager.get_buffer_size("test") == 100
        
        with pytest.raises(KeyError, match="Buffer 'missing' not found"):
            self.buffer_manager.get_buffer_size("missing")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_delete_buffer(self, mock_gl):
        """Test buffer deletion."""
        self.buffer_manager._buffers["test"] = 42
        self.buffer_manager._buffer_sizes["test"] = 100
        
        self.buffer_manager.delete_buffer("test")
        
        mock_gl.glDeleteBuffers.assert_called_with(1, [42])
        assert "test" not in self.buffer_manager._buffers
        assert "test" not in self.buffer_manager._buffer_sizes
        
        with pytest.raises(KeyError, match="Buffer 'test' not found"):
            self.buffer_manager.delete_buffer("test")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_delete_vao(self, mock_gl):
        """Test VAO deletion."""
        self.buffer_manager._vaos["test"] = 123
        
        self.buffer_manager.delete_vao("test")
        
        mock_gl.glDeleteVertexArrays.assert_called_with(1, [123])
        assert "test" not in self.buffer_manager._vaos
        
        with pytest.raises(KeyError, match="VAO 'test' not found"):
            self.buffer_manager.delete_vao("test")
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_cleanup(self, mock_gl):
        """Test cleanup of all resources."""
        # Add some buffers and VAOs
        self.buffer_manager._buffers = {"buf1": 1, "buf2": 2}
        self.buffer_manager._buffer_sizes = {"buf1": 10, "buf2": 20}
        self.buffer_manager._vaos = {"vao1": 10, "vao2": 20}
        
        # Cleanup
        self.buffer_manager.cleanup()
        
        # Verify deletions
        mock_gl.glDeleteBuffers.assert_called_with(2, [1, 2])
        mock_gl.glDeleteVertexArrays.assert_called_with(2, [10, 20])
        
        # Verify state reset
        assert len(self.buffer_manager._buffers) == 0
        assert len(self.buffer_manager._buffer_sizes) == 0
        assert len(self.buffer_manager._vaos) == 0
    
    @patch('gpu_rendering.buffer_manager.gl')
    def test_cleanup_empty(self, mock_gl):
        """Test cleanup with no resources."""
        self.buffer_manager.cleanup()
        
        # Should not call delete functions
        mock_gl.glDeleteBuffers.assert_not_called()
        mock_gl.glDeleteVertexArrays.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])