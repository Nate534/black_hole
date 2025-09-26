"""
Tests for ComputeRenderer class.

Tests compute shader dispatch, GPU synchronization, and performance benchmarking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from gpu_rendering.compute_renderer import ComputeRenderer
from gpu_rendering.shader_manager import ShaderManager
from gpu_rendering.buffer_manager import BufferManager
from physics.black_hole import BlackHole
from physics.particle import Particle
from physics.vector3 import Vector3


class TestComputeRenderer:
    """Test cases for ComputeRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shader_manager = Mock(spec=ShaderManager)
        self.buffer_manager = Mock(spec=BufferManager)
        self.compute_renderer = ComputeRenderer(self.shader_manager, self.buffer_manager)
        
        # Create test black hole
        self.test_black_hole = BlackHole(
            mass=1e30,
            position=Vector3(0.0, 0.0, 0.0)
        )
        
        # Create test particles
        self.test_particles = [
            Particle(mass=1.0, position=Vector3(1.0, 0.0, 0.0), velocity=Vector3(0.0, 1.0, 0.0)),
            Particle(mass=2.0, position=Vector3(-1.0, 0.0, 0.0), velocity=Vector3(0.0, -1.0, 0.0))
        ]
    
    def test_initialization(self):
        """Test ComputeRenderer initialization."""
        assert self.compute_renderer.shader_manager is self.shader_manager
        assert self.compute_renderer.buffer_manager is self.buffer_manager
        assert self.compute_renderer._compute_program is None
        assert self.compute_renderer._work_group_size == 64
    
    def test_load_compute_shader(self):
        """Test loading compute shader."""
        self.shader_manager.load_compute_shader.return_value = 42
        
        self.compute_renderer.load_compute_shader("test_shader.glsl", "test_program")
        
        self.shader_manager.load_compute_shader.assert_called_with("test_shader.glsl", "test_program")
        assert self.compute_renderer._compute_program == 42
    
    def test_setup_compute_uniforms(self):
        """Test setting up compute shader uniforms."""
        self.compute_renderer._compute_program = 42
        
        # Mock black hole methods
        with patch.object(self.test_black_hole, 'get_schwarzschild_radius', return_value=2.95e3):
            with patch.object(self.test_black_hole, 'get_photon_sphere_radius', return_value=4.43e3):
                self.compute_renderer.setup_compute_uniforms(
                    self.test_black_hole, 
                    dt=0.016, 
                    particle_count=100,
                    gravitational_constant=6.67430e-11
                )
        
        # Verify shader program was used
        self.shader_manager.use_program.assert_called_with(42)
        
        # Verify uniform calls
        expected_uniform_calls = [
            ("u_blackHoleMass", self.test_black_hole.mass),
            ("u_blackHolePosition", [0.0, 0.0, 0.0]),
            ("u_deltaTime", 0.016),
            ("u_particleCount", 100),
            ("u_gravitationalConstant", 6.67430e-11),
            ("u_schwarzschildRadius", 2.95e3),
            ("u_photonSphereRadius", 4.43e3)
        ]
        
        for name, value in expected_uniform_calls:
            self.shader_manager.set_uniform.assert_any_call(name, value)
    
    def test_setup_compute_uniforms_no_program(self):
        """Test error when setting uniforms without loaded program."""
        with pytest.raises(RuntimeError, match="No compute shader program loaded"):
            self.compute_renderer.setup_compute_uniforms(self.test_black_hole, 0.016, 100)
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_dispatch_particle_update(self, mock_gl):
        """Test dispatching compute shader for particle updates."""
        self.compute_renderer._compute_program = 42
        particle_count = 150  # Should require 3 work groups (150 / 64 = 2.34, rounded up to 3)
        
        self.compute_renderer.dispatch_particle_update(particle_count, "test_buffer")
        
        # Verify shader program was used
        self.shader_manager.use_program.assert_called_with(42)
        
        # Verify buffer was bound
        self.buffer_manager.bind_buffer.assert_called_with("test_buffer", 0)
        
        # Verify compute dispatch (3 work groups for 150 particles with work group size 64)
        mock_gl.glDispatchCompute.assert_called_with(3, 1, 1)
        
        # Verify memory barrier
        mock_gl.glMemoryBarrier.assert_called_with(mock_gl.GL_SHADER_STORAGE_BARRIER_BIT)
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_dispatch_particle_update_zero_particles(self, mock_gl):
        """Test dispatching with zero particles."""
        self.compute_renderer._compute_program = 42
        
        self.compute_renderer.dispatch_particle_update(0)
        
        # Should not dispatch anything
        mock_gl.glDispatchCompute.assert_not_called()
        mock_gl.glMemoryBarrier.assert_not_called()
    
    def test_dispatch_particle_update_no_program(self):
        """Test error when dispatching without loaded program."""
        with pytest.raises(RuntimeError, match="No compute shader program loaded"):
            self.compute_renderer.dispatch_particle_update(100)
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_synchronize_gpu(self, mock_gl):
        """Test GPU synchronization."""
        self.compute_renderer.synchronize_gpu()
        
        mock_gl.glFinish.assert_called_once()
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_dispatch_and_sync(self, mock_gl):
        """Test combined dispatch and sync operation."""
        self.compute_renderer._compute_program = 42
        
        self.compute_renderer.dispatch_and_sync(64, "test_buffer")
        
        # Verify dispatch was called
        self.shader_manager.use_program.assert_called_with(42)
        self.buffer_manager.bind_buffer.assert_called_with("test_buffer", 0)
        mock_gl.glDispatchCompute.assert_called_with(1, 1, 1)  # 64 particles = 1 work group
        
        # Verify synchronization
        mock_gl.glMemoryBarrier.assert_called_with(mock_gl.GL_SHADER_STORAGE_BARRIER_BIT)
        mock_gl.glFinish.assert_called_once()
    
    def test_set_work_group_size_valid(self):
        """Test setting valid work group sizes."""
        valid_sizes = [32, 64, 128, 256]
        
        for size in valid_sizes:
            self.compute_renderer.set_work_group_size(size)
            assert self.compute_renderer.get_work_group_size() == size
    
    def test_set_work_group_size_invalid(self):
        """Test error handling for invalid work group sizes."""
        invalid_sizes = [0, -1, 33, 100, 129]  # Not powers of 2 or non-positive
        
        for size in invalid_sizes:
            with pytest.raises(ValueError, match="Work group size must be a positive power of 2"):
                self.compute_renderer.set_work_group_size(size)
    
    def test_get_work_group_size(self):
        """Test getting work group size."""
        assert self.compute_renderer.get_work_group_size() == 64  # Default
        
        self.compute_renderer.set_work_group_size(128)
        assert self.compute_renderer.get_work_group_size() == 128
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_max_work_group_size(self, mock_gl):
        """Test querying maximum work group size."""
        mock_gl.glGetIntegerv.return_value = [1024]
        
        max_size = self.compute_renderer.get_max_work_group_size()
        
        mock_gl.glGetIntegerv.assert_called_with(mock_gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE)
        assert max_size == 1024
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_max_work_group_size_single_value(self, mock_gl):
        """Test querying maximum work group size when single value returned."""
        mock_gl.glGetIntegerv.return_value = 512
        
        max_size = self.compute_renderer.get_max_work_group_size()
        
        assert max_size == 512
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_max_work_groups(self, mock_gl):
        """Test querying maximum work groups."""
        mock_gl.glGetIntegerv.return_value = [65535, 65535, 65535]
        
        max_groups = self.compute_renderer.get_max_work_groups()
        
        mock_gl.glGetIntegerv.assert_called_with(mock_gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT)
        assert max_groups == (65535, 65535, 65535)
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_max_work_groups_fallback(self, mock_gl):
        """Test fallback when work group query fails."""
        mock_gl.glGetIntegerv.return_value = None
        
        max_groups = self.compute_renderer.get_max_work_groups()
        
        assert max_groups == (65535, 65535, 65535)
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_check_compute_shader_support_version_43(self, mock_gl):
        """Test compute shader support check with OpenGL 4.3+."""
        mock_gl.glGetString.return_value = b"4.3.0 NVIDIA 460.89"
        
        supported = self.compute_renderer.check_compute_shader_support()
        
        assert supported is True
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_check_compute_shader_support_version_45(self, mock_gl):
        """Test compute shader support check with OpenGL 4.5+."""
        mock_gl.glGetString.return_value = b"4.5.0 Core Profile Context"
        
        supported = self.compute_renderer.check_compute_shader_support()
        
        assert supported is True
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_check_compute_shader_support_extension(self, mock_gl):
        """Test compute shader support check with extension."""
        mock_gl.glGetString.side_effect = [
            b"4.2.0 Core Profile Context",  # Version call
            b"GL_ARB_compute_shader GL_ARB_shader_storage_buffer_object"  # Extensions call
        ]
        
        supported = self.compute_renderer.check_compute_shader_support()
        
        assert supported is True
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_check_compute_shader_support_not_supported(self, mock_gl):
        """Test compute shader support check when not supported."""
        mock_gl.glGetString.side_effect = [
            b"3.3.0 Core Profile Context",  # Version call
            b"GL_ARB_vertex_buffer_object"  # Extensions call (no compute shader)
        ]
        
        supported = self.compute_renderer.check_compute_shader_support()
        
        assert supported is False
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_check_compute_shader_support_exception(self, mock_gl):
        """Test compute shader support check with exception."""
        mock_gl.glGetString.side_effect = Exception("OpenGL error")
        
        supported = self.compute_renderer.check_compute_shader_support()
        
        assert supported is False
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_compute_shader_limits(self, mock_gl):
        """Test getting compute shader limits."""
        # Mock various limit queries
        def mock_get_integer(param):
            limits_map = {
                mock_gl.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: 1024,
                mock_gl.GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: 32768,
                mock_gl.GL_MAX_COMPUTE_UNIFORM_BLOCKS: 14,
                mock_gl.GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS: 16,
                mock_gl.GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS: 8,
                mock_gl.GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: 16
            }
            return limits_map.get(param, 0)
        
        mock_gl.glGetIntegerv.side_effect = mock_get_integer
        
        # Mock the methods that get_compute_shader_limits calls
        with patch.object(self.compute_renderer, 'get_max_work_groups', return_value=(65535, 65535, 65535)):
            with patch.object(self.compute_renderer, 'get_max_work_group_size', return_value=1024):
                limits = self.compute_renderer.get_compute_shader_limits()
        
        expected_limits = {
            'max_work_group_count': (65535, 65535, 65535),
            'max_work_group_size': 1024,
            'max_work_group_invocations': 1024,
            'max_shared_memory_size': 32768,
            'max_uniform_blocks': 14,
            'max_texture_image_units': 16,
            'max_atomic_counter_buffers': 8,
            'max_shader_storage_blocks': 16
        }
        
        for key, value in expected_limits.items():
            assert limits[key] == value
    
    @patch('gpu_rendering.compute_renderer.gl')
    def test_get_compute_shader_limits_exception(self, mock_gl):
        """Test getting compute shader limits with exception."""
        mock_gl.glGetIntegerv.side_effect = Exception("OpenGL error")
        
        limits = self.compute_renderer.get_compute_shader_limits()
        
        assert 'error' in limits
        assert "OpenGL error" in limits['error']
    
    @patch('time.perf_counter')
    @patch('gpu_rendering.compute_renderer.gl')
    def test_benchmark_compute_performance(self, mock_gl, mock_time):
        """Test compute performance benchmarking."""
        self.compute_renderer._compute_program = 42
        
        # Mock time progression - simple start and end times
        mock_time.side_effect = [0.0, 1.0]  # Start and end of benchmark
        
        # Mock buffer manager methods
        self.buffer_manager.create_particle_buffer.return_value = 123
        
        result = self.compute_renderer.benchmark_compute_performance(1000, iterations=50)
        
        # Verify buffer operations
        self.buffer_manager.create_particle_buffer.assert_called_once()
        self.buffer_manager.delete_buffer.assert_called_with("benchmark_buffer")
        
        # Verify results structure and reasonable values
        assert result['particle_count'] == 1000
        assert result['iterations'] == 50
        assert 'total_time_seconds' in result
        assert 'avg_time_per_iteration_seconds' in result
        assert 'particles_per_second' in result
        assert result['work_group_size'] == 64
        
        # Verify values are positive and reasonable
        assert result['total_time_seconds'] > 0
        assert result['avg_time_per_iteration_seconds'] > 0
        assert result['particles_per_second'] > 0
    
    def test_benchmark_compute_performance_no_program(self):
        """Test benchmarking error when no program loaded."""
        with pytest.raises(RuntimeError, match="No compute shader program loaded"):
            self.compute_renderer.benchmark_compute_performance(100)
    
    def test_cleanup(self):
        """Test cleanup method."""
        self.compute_renderer._compute_program = 42
        
        self.compute_renderer.cleanup()
        
        assert self.compute_renderer._compute_program is None


if __name__ == "__main__":
    pytest.main([__file__])