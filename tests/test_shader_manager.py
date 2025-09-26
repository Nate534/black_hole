"""
Tests for ShaderManager class.

Tests shader loading, compilation, and program management functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from gpu_rendering.shader_manager import ShaderManager, ShaderCompilationError


class TestShaderManager:
    """Test cases for ShaderManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shader_manager = ShaderManager()
        
        # Mock OpenGL functions
        self.gl_mock = Mock()
        
        # Sample shader sources
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main() {
            gl_Position = vec4(aPos, 1.0);
        }
        """
        
        self.fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.5, 0.2, 1.0);
        }
        """
        
        self.compute_shader_source = """
        #version 430
        layout(local_size_x = 64) in;
        layout(std430, binding = 0) buffer ParticleBuffer {
            vec4 particles[];
        };
        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= particles.length()) return;
            particles[index].xyz += vec3(0.1, 0.0, 0.0);
        }
        """
    
    def create_temp_shader_file(self, content: str) -> str:
        """Create a temporary shader file with given content."""
        fd, path = tempfile.mkstemp(suffix='.glsl')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_load_compute_shader_success(self, mock_gl):
        """Test successful compute shader loading and compilation."""
        # Setup mocks
        mock_gl.glCreateShader.return_value = 1
        mock_gl.glGetShaderiv.return_value = True  # Compilation success
        mock_gl.glCreateProgram.return_value = 2
        mock_gl.glGetProgramiv.return_value = True  # Linking success
        
        # Create temporary shader file
        shader_path = self.create_temp_shader_file(self.compute_shader_source)
        
        try:
            # Test loading
            program_id = self.shader_manager.load_compute_shader(shader_path, "test_compute")
            
            # Verify calls
            mock_gl.glCreateShader.assert_called_with(mock_gl.GL_COMPUTE_SHADER)
            mock_gl.glShaderSource.assert_called_once()
            mock_gl.glCompileShader.assert_called_once()
            mock_gl.glCreateProgram.assert_called_once()
            mock_gl.glAttachShader.assert_called_with(2, 1)
            mock_gl.glLinkProgram.assert_called_with(2)
            mock_gl.glDeleteShader.assert_called_with(1)
            
            # Verify return value and storage
            assert program_id == 2
            assert self.shader_manager.get_program_id("test_compute") == 2
            
        finally:
            os.unlink(shader_path)
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_load_compute_shader_compilation_error(self, mock_gl):
        """Test compute shader compilation error handling."""
        # Setup mocks for compilation failure
        mock_gl.glCreateShader.return_value = 1
        mock_gl.glGetShaderiv.return_value = False  # Compilation failure
        mock_gl.glGetShaderInfoLog.return_value = b"Syntax error in shader"
        
        # Create temporary shader file
        shader_path = self.create_temp_shader_file("invalid shader code")
        
        try:
            # Test compilation error
            with pytest.raises(ShaderCompilationError, match="Compute shader compilation failed"):
                self.shader_manager.load_compute_shader(shader_path)
            
            # Verify cleanup
            mock_gl.glDeleteShader.assert_called_with(1)
            
        finally:
            os.unlink(shader_path)
    
    def test_load_compute_shader_file_not_found(self):
        """Test error handling when shader file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.shader_manager.load_compute_shader("nonexistent.glsl")
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_load_vertex_fragment_shaders_success(self, mock_gl):
        """Test successful vertex and fragment shader loading."""
        # Setup mocks
        mock_gl.glCreateShader.side_effect = [1, 2]  # vertex, fragment
        mock_gl.glGetShaderiv.return_value = True  # Compilation success
        mock_gl.glCreateProgram.return_value = 3
        mock_gl.glGetProgramiv.return_value = True  # Linking success
        
        # Create temporary shader files
        vertex_path = self.create_temp_shader_file(self.vertex_shader_source)
        fragment_path = self.create_temp_shader_file(self.fragment_shader_source)
        
        try:
            # Test loading
            program_id = self.shader_manager.load_vertex_fragment_shaders(
                vertex_path, fragment_path, "test_render"
            )
            
            # Verify calls
            assert mock_gl.glCreateShader.call_count == 2
            mock_gl.glCreateShader.assert_any_call(mock_gl.GL_VERTEX_SHADER)
            mock_gl.glCreateShader.assert_any_call(mock_gl.GL_FRAGMENT_SHADER)
            assert mock_gl.glShaderSource.call_count == 2
            assert mock_gl.glCompileShader.call_count == 2
            mock_gl.glAttachShader.assert_any_call(3, 1)
            mock_gl.glAttachShader.assert_any_call(3, 2)
            mock_gl.glLinkProgram.assert_called_with(3)
            
            # Verify return value and storage
            assert program_id == 3
            assert self.shader_manager.get_program_id("test_render") == 3
            
        finally:
            os.unlink(vertex_path)
            os.unlink(fragment_path)
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_load_vertex_fragment_shaders_linking_error(self, mock_gl):
        """Test program linking error handling."""
        # Setup mocks
        mock_gl.glCreateShader.side_effect = [1, 2]
        mock_gl.glGetShaderiv.return_value = True  # Compilation success
        mock_gl.glCreateProgram.return_value = 3
        mock_gl.glGetProgramiv.return_value = False  # Linking failure
        mock_gl.glGetProgramInfoLog.return_value = b"Linking error"
        
        # Create temporary shader files
        vertex_path = self.create_temp_shader_file(self.vertex_shader_source)
        fragment_path = self.create_temp_shader_file(self.fragment_shader_source)
        
        try:
            # Test linking error
            with pytest.raises(ShaderCompilationError, match="Program linking failed"):
                self.shader_manager.load_vertex_fragment_shaders(vertex_path, fragment_path)
            
            # Verify cleanup
            mock_gl.glDeleteShader.assert_any_call(1)
            mock_gl.glDeleteShader.assert_any_call(2)
            mock_gl.glDeleteProgram.assert_called_with(3)
            
        finally:
            os.unlink(vertex_path)
            os.unlink(fragment_path)
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_use_program(self, mock_gl):
        """Test program binding functionality."""
        # Test initial binding
        self.shader_manager.use_program(5)
        mock_gl.glUseProgram.assert_called_with(5)
        
        # Test that same program isn't bound twice
        mock_gl.reset_mock()
        self.shader_manager.use_program(5)
        mock_gl.glUseProgram.assert_not_called()
        
        # Test binding different program
        self.shader_manager.use_program(10)
        mock_gl.glUseProgram.assert_called_with(10)
    
    def test_use_program_by_name(self):
        """Test program binding by name."""
        # Add a program to the manager
        self.shader_manager._programs["test"] = 42
        
        with patch('gpu_rendering.shader_manager.gl') as mock_gl:
            self.shader_manager.use_program_by_name("test")
            mock_gl.glUseProgram.assert_called_with(42)
        
        # Test error for non-existent program
        with pytest.raises(KeyError, match="Shader program 'nonexistent' not found"):
            self.shader_manager.use_program_by_name("nonexistent")
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_set_uniform_various_types(self, mock_gl):
        """Test setting uniform variables of different types."""
        # Setup current program
        self.shader_manager._current_program = 1
        mock_gl.glGetUniformLocation.return_value = 0  # Valid location
        
        # Test different uniform types
        test_cases = [
            (True, mock_gl.glUniform1i, (0, 1)),
            (False, mock_gl.glUniform1i, (0, 0)),
            (42, mock_gl.glUniform1i, (0, 42)),
            (3.14, mock_gl.glUniform1f, (0, 3.14)),
            ([1.0, 2.0], mock_gl.glUniform2f, (0, 1.0, 2.0)),
            ([1.0, 2.0, 3.0], mock_gl.glUniform3f, (0, 1.0, 2.0, 3.0)),
            ([1.0, 2.0, 3.0, 4.0], mock_gl.glUniform4f, (0, 1.0, 2.0, 3.0, 4.0)),
        ]
        
        for value, expected_func, expected_args in test_cases:
            mock_gl.reset_mock()
            mock_gl.glGetUniformLocation.return_value = 0
            
            self.shader_manager.set_uniform("test_uniform", value)
            expected_func.assert_called_with(*expected_args)
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_set_uniform_matrix(self, mock_gl):
        """Test setting matrix uniform."""
        self.shader_manager._current_program = 1
        mock_gl.glGetUniformLocation.return_value = 0
        
        # Test 4x4 matrix
        matrix = [1.0] * 16
        self.shader_manager.set_uniform("matrix", matrix)
        mock_gl.glUniformMatrix4fv.assert_called_with(0, 1, mock_gl.GL_FALSE, matrix)
    
    def test_set_uniform_no_program(self):
        """Test error when no program is bound."""
        with pytest.raises(RuntimeError, match="No shader program is currently bound"):
            self.shader_manager.set_uniform("test", 1.0)
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_set_uniform_invalid_location(self, mock_gl):
        """Test handling of invalid uniform location."""
        self.shader_manager._current_program = 1
        mock_gl.glGetUniformLocation.return_value = -1  # Invalid location
        
        # Should not raise error, just return silently
        self.shader_manager.set_uniform("nonexistent", 1.0)
        mock_gl.glUniform1f.assert_not_called()
    
    def test_set_uniform_invalid_types(self):
        """Test error handling for invalid uniform types."""
        self.shader_manager._current_program = 1
        
        with patch('gpu_rendering.shader_manager.gl') as mock_gl:
            mock_gl.glGetUniformLocation.return_value = 0
            
            # Test unsupported array size
            with pytest.raises(ValueError, match="Unsupported uniform array size"):
                self.shader_manager.set_uniform("test", [1, 2, 3, 4, 5])
            
            # Test unsupported type
            with pytest.raises(TypeError, match="Unsupported uniform type"):
                self.shader_manager.set_uniform("test", {"invalid": "type"})
    
    def test_get_program_id(self):
        """Test getting program ID by name."""
        self.shader_manager._programs["test"] = 123
        assert self.shader_manager.get_program_id("test") == 123
        
        with pytest.raises(KeyError, match="Shader program 'missing' not found"):
            self.shader_manager.get_program_id("missing")
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_delete_program(self, mock_gl):
        """Test program deletion."""
        # Add program and set as current
        self.shader_manager._programs["test"] = 42
        self.shader_manager._current_program = 42
        
        # Delete program
        self.shader_manager.delete_program("test")
        
        # Verify cleanup
        mock_gl.glUseProgram.assert_called_with(0)
        mock_gl.glDeleteProgram.assert_called_with(42)
        assert "test" not in self.shader_manager._programs
        assert self.shader_manager._current_program is None
        
        # Test error for non-existent program
        with pytest.raises(KeyError, match="Shader program 'missing' not found"):
            self.shader_manager.delete_program("missing")
    
    @patch('gpu_rendering.shader_manager.gl')
    def test_cleanup(self, mock_gl):
        """Test cleanup of all programs."""
        # Add some programs
        self.shader_manager._programs = {"prog1": 1, "prog2": 2, "prog3": 3}
        self.shader_manager._current_program = 2
        
        # Cleanup
        self.shader_manager.cleanup()
        
        # Verify all programs deleted
        assert mock_gl.glDeleteProgram.call_count == 3
        mock_gl.glDeleteProgram.assert_any_call(1)
        mock_gl.glDeleteProgram.assert_any_call(2)
        mock_gl.glDeleteProgram.assert_any_call(3)
        mock_gl.glUseProgram.assert_called_with(0)
        
        # Verify state reset
        assert len(self.shader_manager._programs) == 0
        assert self.shader_manager._current_program is None


if __name__ == "__main__":
    pytest.main([__file__])