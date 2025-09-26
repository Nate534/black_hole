"""
BufferManager for GPU data management in the black hole simulation.

This module manages OpenGL buffer objects and vertex array objects for efficient
GPU data transfer and rendering operations.
"""

import numpy as np
from typing import List, Dict, Optional
from OpenGL import GL as gl
from physics.particle import Particle


class BufferAllocationError(Exception):
    """Exception raised when buffer allocation fails."""
    pass


class BufferManager:
    """
    Manages OpenGL buffer objects and vertex array objects.
    
    Provides methods for creating, updating, and managing GPU buffers
    for particle data and vertex array objects for rendering.
    """
    
    def __init__(self):
        """Initialize the buffer manager."""
        self._buffers: Dict[str, int] = {}  # Buffer name -> OpenGL buffer ID
        self._buffer_sizes: Dict[str, int] = {}  # Buffer name -> particle count
        self._vaos: Dict[str, int] = {}  # VAO name -> OpenGL VAO ID
        
        # Default buffer size for empty particle lists
        self._default_buffer_size = 1000
    
    def _particles_to_array(self, particles: List[Particle]) -> np.ndarray:
        """
        Convert a list of particles to a numpy array for GPU upload.
        
        Array layout per particle (8 floats, 32 bytes):
        - Position: x, y, z (3 floats, 12 bytes)
        - Velocity: x, y, z (3 floats, 12 bytes)  
        - Mass: mass (1 float, 4 bytes)
        - Active: active flag (1 float, 4 bytes)
        
        Args:
            particles: List of particles to convert
            
        Returns:
            np.ndarray: Array of particle data with shape (num_particles, 8)
        """
        if not particles:
            return np.empty((0, 8), dtype=np.float32)
        
        # Create array with shape (num_particles, 8)
        particle_array = np.zeros((len(particles), 8), dtype=np.float32)
        
        for i, particle in enumerate(particles):
            # Position (3 floats)
            particle_array[i, 0] = particle.position.x
            particle_array[i, 1] = particle.position.y
            particle_array[i, 2] = particle.position.z
            
            # Velocity (3 floats)
            particle_array[i, 3] = particle.velocity.x
            particle_array[i, 4] = particle.velocity.y
            particle_array[i, 5] = particle.velocity.z
            
            # Mass (1 float)
            particle_array[i, 6] = particle.mass
            
            # Active flag (1 float, 1.0 for active, 0.0 for inactive)
            particle_array[i, 7] = 1.0 if particle.active else 0.0
        
        return particle_array
    
    def create_particle_buffer(self, particles: List[Particle], name: Optional[str] = None) -> int:
        """
        Create a new OpenGL buffer for particle data.
        
        Args:
            particles: List of particles to store in buffer
            name: Optional name for the buffer (for tracking)
            
        Returns:
            int: OpenGL buffer ID
            
        Raises:
            BufferAllocationError: If buffer creation fails
        """
        # Generate OpenGL buffer
        buffer_id = gl.glGenBuffers(1)
        if buffer_id == 0:
            raise BufferAllocationError("Failed to generate OpenGL buffer")
        
        try:
            # Bind buffer
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
            
            # Convert particles to array
            particle_array = self._particles_to_array(particles)
            
            # Determine buffer size
            if len(particles) > 0:
                buffer_size = len(particles)
                data = particle_array.flatten()
            else:
                # Create buffer with default size for empty particle list
                buffer_size = self._default_buffer_size
                data = np.zeros(buffer_size * 8, dtype=np.float32)
            
            # Upload data to GPU
            gl.glBufferData(
                gl.GL_SHADER_STORAGE_BUFFER,
                data.nbytes,
                data,
                gl.GL_DYNAMIC_DRAW
            )
            
            # Unbind buffer
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            
            # Check for OpenGL errors
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                gl.glDeleteBuffers(1, [buffer_id])
                raise BufferAllocationError(f"OpenGL error during buffer creation: {error}")
            
            # Store buffer info if name provided
            if name is not None:
                self._buffers[name] = buffer_id
                self._buffer_sizes[name] = buffer_size
            
            return buffer_id
            
        except Exception as e:
            # Clean up on failure
            gl.glDeleteBuffers(1, [buffer_id])
            raise e
    
    def update_particle_buffer(self, name: str, particles: List[Particle]) -> None:
        """
        Update an existing particle buffer with new data.
        
        Args:
            name: Name of the buffer to update
            particles: New particle data
            
        Raises:
            KeyError: If buffer name not found
            BufferAllocationError: If update fails
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found")
        
        buffer_id = self._buffers[name]
        current_size = self._buffer_sizes[name]
        
        # Convert particles to array
        particle_array = self._particles_to_array(particles)
        particle_count = len(particles)
        
        # Bind buffer
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        
        try:
            if particle_count <= current_size:
                # Buffer is large enough, use glBufferSubData for efficiency
                if particle_count > 0:
                    data = particle_array.flatten()
                    gl.glBufferSubData(
                        gl.GL_SHADER_STORAGE_BUFFER,
                        0,  # offset
                        data.nbytes,
                        data
                    )
            else:
                # Buffer too small, reallocate with new size
                new_size = max(particle_count, int(current_size * 1.5))
                
                if particle_count > 0:
                    data = particle_array.flatten()
                    # Pad with zeros if needed
                    if new_size > particle_count:
                        padded_data = np.zeros(new_size * 8, dtype=np.float32)
                        padded_data[:len(data)] = data
                        data = padded_data
                else:
                    data = np.zeros(new_size * 8, dtype=np.float32)
                
                gl.glBufferData(
                    gl.GL_SHADER_STORAGE_BUFFER,
                    data.nbytes,
                    data,
                    gl.GL_DYNAMIC_DRAW
                )
                
                # Update stored size
                self._buffer_sizes[name] = new_size
            
            # Unbind buffer
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            
            # Check for OpenGL errors
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                raise BufferAllocationError(f"OpenGL error during buffer update: {error}")
                
        except Exception as e:
            # Unbind buffer on error
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            raise e
    
    def create_vertex_array(self, name: Optional[str] = None) -> int:
        """
        Create a new OpenGL vertex array object.
        
        Args:
            name: Optional name for the VAO (for tracking)
            
        Returns:
            int: OpenGL VAO ID
            
        Raises:
            BufferAllocationError: If VAO creation fails
        """
        vao_id = gl.glGenVertexArrays(1)
        if vao_id == 0:
            raise BufferAllocationError("Failed to generate OpenGL vertex array")
        
        # Store VAO info if name provided
        if name is not None:
            self._vaos[name] = vao_id
        
        return vao_id
    
    def setup_particle_vertex_attributes(self, vao_name: str, buffer_name: str) -> None:
        """
        Setup vertex attributes for particle rendering.
        
        Configures vertex attributes for the particle buffer layout:
        - Location 0: Position (3 floats)
        - Location 1: Velocity (3 floats)
        - Location 2: Mass (1 float)
        - Location 3: Active (1 float)
        
        Args:
            vao_name: Name of the VAO to configure
            buffer_name: Name of the buffer containing particle data
            
        Raises:
            KeyError: If VAO or buffer name not found
        """
        if vao_name not in self._vaos:
            raise KeyError(f"VAO '{vao_name}' not found")
        if buffer_name not in self._buffers:
            raise KeyError(f"Buffer '{buffer_name}' not found")
        
        vao_id = self._vaos[vao_name]
        buffer_id = self._buffers[buffer_name]
        
        # Bind VAO
        gl.glBindVertexArray(vao_id)
        
        # Bind buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
        
        # Stride: 8 floats * 4 bytes = 32 bytes per particle
        stride = 8 * 4
        
        # Position attribute (location 0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(0))
        gl.glEnableVertexAttribArray(0)
        
        # Velocity attribute (location 1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(12))
        gl.glEnableVertexAttribArray(1)
        
        # Mass attribute (location 2)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(24))
        gl.glEnableVertexAttribArray(2)
        
        # Active attribute (location 3)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(28))
        gl.glEnableVertexAttribArray(3)
        
        # Unbind
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)
    
    def bind_buffer(self, name: str, binding_point: int = 0) -> None:
        """
        Bind a buffer to a shader storage buffer binding point.
        
        Args:
            name: Name of the buffer to bind
            binding_point: Shader storage buffer binding point (default: 0)
            
        Raises:
            KeyError: If buffer name not found
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found")
        
        buffer_id = self._buffers[name]
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding_point, buffer_id)
    
    def bind_vertex_array(self, name: str) -> None:
        """
        Bind a vertex array object.
        
        Args:
            name: Name of the VAO to bind
            
        Raises:
            KeyError: If VAO name not found
        """
        if name not in self._vaos:
            raise KeyError(f"VAO '{name}' not found")
        
        vao_id = self._vaos[name]
        gl.glBindVertexArray(vao_id)
    
    def unbind_vertex_array(self) -> None:
        """Unbind the currently bound vertex array object."""
        gl.glBindVertexArray(0)
    
    def get_buffer_id(self, name: str) -> int:
        """
        Get the OpenGL buffer ID for a named buffer.
        
        Args:
            name: Name of the buffer
            
        Returns:
            int: OpenGL buffer ID
            
        Raises:
            KeyError: If buffer name not found
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found")
        return self._buffers[name]
    
    def get_vao_id(self, name: str) -> int:
        """
        Get the OpenGL VAO ID for a named VAO.
        
        Args:
            name: Name of the VAO
            
        Returns:
            int: OpenGL VAO ID
            
        Raises:
            KeyError: If VAO name not found
        """
        if name not in self._vaos:
            raise KeyError(f"VAO '{name}' not found")
        return self._vaos[name]
    
    def get_buffer_size(self, name: str) -> int:
        """
        Get the particle capacity of a named buffer.
        
        Args:
            name: Name of the buffer
            
        Returns:
            int: Number of particles the buffer can hold
            
        Raises:
            KeyError: If buffer name not found
        """
        if name not in self._buffer_sizes:
            raise KeyError(f"Buffer '{name}' not found")
        return self._buffer_sizes[name]
    
    def delete_buffer(self, name: str) -> None:
        """
        Delete a named buffer and remove it from tracking.
        
        Args:
            name: Name of the buffer to delete
            
        Raises:
            KeyError: If buffer name not found
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found")
        
        buffer_id = self._buffers[name]
        gl.glDeleteBuffers(1, [buffer_id])
        
        del self._buffers[name]
        del self._buffer_sizes[name]
    
    def delete_vao(self, name: str) -> None:
        """
        Delete a named VAO and remove it from tracking.
        
        Args:
            name: Name of the VAO to delete
            
        Raises:
            KeyError: If VAO name not found
        """
        if name not in self._vaos:
            raise KeyError(f"VAO '{name}' not found")
        
        vao_id = self._vaos[name]
        gl.glDeleteVertexArrays(1, [vao_id])
        
        del self._vaos[name]
    
    def cleanup(self) -> None:
        """
        Clean up all managed OpenGL resources.
        
        Deletes all buffers and VAOs and clears internal tracking.
        """
        # Delete all buffers
        if self._buffers:
            buffer_ids = list(self._buffers.values())
            gl.glDeleteBuffers(len(buffer_ids), buffer_ids)
        
        # Delete all VAOs
        if self._vaos:
            vao_ids = list(self._vaos.values())
            gl.glDeleteVertexArrays(len(vao_ids), vao_ids)
        
        # Clear tracking dictionaries
        self._buffers.clear()
        self._buffer_sizes.clear()
        self._vaos.clear()