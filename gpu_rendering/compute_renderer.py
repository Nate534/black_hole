"""
ComputeRenderer for GPU-accelerated particle physics calculations.

This module implements GPU compute shader dispatch for particle updates,
GPU-CPU synchronization, and performance monitoring for the black hole simulation.
"""

import time
from typing import Optional, Dict, Any
import OpenGL.GL as gl
from .shader_manager import ShaderManager, ShaderCompilationError
from .buffer_manager import BufferManager
from physics.black_hole import BlackHole
from physics.particle import Particle


class ComputeRenderer:
    """
    Manages GPU compute shader execution for particle physics calculations.
    
    Provides methods for dispatching compute shaders, synchronizing GPU-CPU operations,
    and monitoring performance of GPU computations.
    """
    
    def __init__(self, shader_manager: ShaderManager, buffer_manager: BufferManager):
        """
        Initialize the compute renderer.
        
        Args:
            shader_manager: ShaderManager instance for compute shader management
            buffer_manager: BufferManager instance for GPU buffer operations
        """
        self.shader_manager = shader_manager
        self.buffer_manager = buffer_manager
        
        # Compute shader program ID
        self._particle_update_program: Optional[int] = None
        
        # Performance tracking
        self._last_dispatch_time: float = 0.0
        self._dispatch_count: int = 0
        self._total_dispatch_time: float = 0.0
        
        # GPU synchronization
        self._sync_fence: Optional[int] = None
        
        # Compute shader work group sizes
        self._local_work_group_size = 64  # Particles per work group
        
    def load_particle_update_shader(self, shader_path: str) -> None:
        """
        Load and compile the particle update compute shader.
        
        Args:
            shader_path: Path to the compute shader file
            
        Raises:
            ShaderCompilationError: If shader compilation fails
            FileNotFoundError: If shader file doesn't exist
        """
        try:
            self._particle_update_program = self.shader_manager.load_compute_shader(
                shader_path, "particle_update"
            )
        except (ShaderCompilationError, FileNotFoundError) as e:
            raise e
    
    def setup_compute_uniforms(self, black_hole: BlackHole, dt: float, 
                             num_particles: int) -> None:
        """
        Set up uniform variables for the compute shader.
        
        Args:
            black_hole: BlackHole instance with mass and position
            dt: Time step for physics integration
            num_particles: Number of particles to process
            
        Raises:
            RuntimeError: If particle update shader is not loaded
        """
        if self._particle_update_program is None:
            raise RuntimeError("Particle update shader not loaded")
        
        # Use the particle update shader program
        self.shader_manager.use_program(self._particle_update_program)
        
        # Set black hole properties
        self.shader_manager.set_uniform("u_black_hole_mass", black_hole.mass)
        self.shader_manager.set_uniform("u_black_hole_position", [
            black_hole.position.x, 
            black_hole.position.y, 
            black_hole.position.z
        ])
        
        # Set physics constants
        self.shader_manager.set_uniform("u_gravitational_constant", black_hole.G)
        self.shader_manager.set_uniform("u_speed_of_light", black_hole.c)
        
        # Set integration parameters
        self.shader_manager.set_uniform("u_delta_time", dt)
        self.shader_manager.set_uniform("u_num_particles", num_particles)
        
        # Set derived black hole properties
        schwarzschild_radius = black_hole.get_schwarzschild_radius()
        photon_sphere_radius = black_hole.get_photon_sphere_radius()
        
        self.shader_manager.set_uniform("u_schwarzschild_radius", schwarzschild_radius)
        self.shader_manager.set_uniform("u_photon_sphere_radius", photon_sphere_radius)
    
    def dispatch_particle_update(self, num_particles: int, 
                               particle_buffer_name: str = "particles") -> None:
        """
        Dispatch the compute shader to update particle positions and velocities.
        
        Args:
            num_particles: Number of particles to process
            particle_buffer_name: Name of the particle buffer to bind
            
        Raises:
            RuntimeError: If particle update shader is not loaded
            KeyError: If particle buffer not found
        """
        if self._particle_update_program is None:
            raise RuntimeError("Particle update shader not loaded")
        
        if num_particles <= 0:
            return
        
        # Record start time for performance tracking
        start_time = time.perf_counter()
        
        # Use the particle update shader program
        self.shader_manager.use_program(self._particle_update_program)
        
        # Bind particle buffer to shader storage buffer binding point 0
        self.buffer_manager.bind_buffer(particle_buffer_name, binding_point=0)
        
        # Calculate work group dispatch size
        # Round up to ensure all particles are processed
        work_groups = (num_particles + self._local_work_group_size - 1) // self._local_work_group_size
        
        # Dispatch compute shader
        gl.glDispatchCompute(work_groups, 1, 1)
        
        # Insert memory barrier to ensure compute shader writes are visible
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        
        # Update performance tracking
        end_time = time.perf_counter()
        dispatch_time = end_time - start_time
        
        self._last_dispatch_time = dispatch_time
        self._dispatch_count += 1
        self._total_dispatch_time += dispatch_time
    
    def synchronize_gpu(self, timeout_ns: int = 1000000000) -> bool:
        """
        Synchronize GPU operations and wait for completion.
        
        Uses OpenGL sync objects to ensure all GPU operations complete
        before returning control to the CPU.
        
        Args:
            timeout_ns: Timeout in nanoseconds (default: 1 second)
            
        Returns:
            bool: True if synchronization completed successfully, False if timeout
        """
        # Insert a fence sync object
        self._sync_fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        
        if self._sync_fence is None:
            return False
        
        # Wait for the fence with timeout
        result = gl.glClientWaitSync(
            self._sync_fence, 
            gl.GL_SYNC_FLUSH_COMMANDS_BIT, 
            timeout_ns
        )
        
        # Clean up the fence
        gl.glDeleteSync(self._sync_fence)
        self._sync_fence = None
        
        return result == gl.GL_ALREADY_SIGNALED or result == gl.GL_CONDITION_SATISFIED
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for compute shader execution.
        
        Returns:
            Dict[str, Any]: Dictionary containing performance metrics
        """
        avg_dispatch_time = (self._total_dispatch_time / self._dispatch_count 
                           if self._dispatch_count > 0 else 0.0)
        
        return {
            "last_dispatch_time_ms": self._last_dispatch_time * 1000.0,
            "average_dispatch_time_ms": avg_dispatch_time * 1000.0,
            "total_dispatches": self._dispatch_count,
            "total_compute_time_ms": self._total_dispatch_time * 1000.0,
            "dispatches_per_second": (self._dispatch_count / self._total_dispatch_time 
                                    if self._total_dispatch_time > 0 else 0.0)
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self._last_dispatch_time = 0.0
        self._dispatch_count = 0
        self._total_dispatch_time = 0.0
    
    def set_local_work_group_size(self, size: int) -> None:
        """
        Set the local work group size for compute shader dispatch.
        
        Args:
            size: Work group size (should be a power of 2, typically 32, 64, or 128)
            
        Raises:
            ValueError: If size is not positive
        """
        if size <= 0:
            raise ValueError("Work group size must be positive")
        
        self._local_work_group_size = size
    
    def get_local_work_group_size(self) -> int:
        """
        Get the current local work group size.
        
        Returns:
            int: Current work group size
        """
        return self._local_work_group_size
    
    def check_compute_shader_support(self) -> bool:
        """
        Check if compute shaders are supported by the current OpenGL context.
        
        Returns:
            bool: True if compute shaders are supported
        """
        # Check OpenGL version (compute shaders require OpenGL 4.3+)
        version_string = gl.glGetString(gl.GL_VERSION).decode()
        
        try:
            # Parse version string (format: "major.minor.patch ...")
            version_parts = version_string.split()[0].split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            
            # Compute shaders require OpenGL 4.3 or higher
            return (major > 4) or (major == 4 and minor >= 3)
        except (IndexError, ValueError):
            return False
    
    def get_max_work_group_size(self) -> tuple[int, int, int]:
        """
        Get the maximum work group size supported by the GPU.
        
        Returns:
            tuple[int, int, int]: Maximum work group size in (x, y, z) dimensions
        """
        max_size = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0), \
                   gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1), \
                   gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2)
        return max_size
    
    def get_max_work_group_invocations(self) -> int:
        """
        Get the maximum number of work group invocations.
        
        Returns:
            int: Maximum work group invocations
        """
        return gl.glGetIntegerv(gl.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
    
    def cleanup(self) -> None:
        """
        Clean up compute renderer resources.
        
        Cleans up any remaining sync objects and resets state.
        """
        if self._sync_fence is not None:
            gl.glDeleteSync(self._sync_fence)
            self._sync_fence = None
        
        self._particle_update_program = None
        self.reset_performance_stats()
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            self.cleanup()
        except:
            # Ignore errors during cleanup (OpenGL context might be gone)
            pass