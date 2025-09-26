"""
ShaderManager for OpenGL shader program management.

Handles loading, compilation, and management of vertex, fragment, and compute shaders.
"""

import OpenGL.GL as gl
from typing import Dict, Any, Optional
import os


class ShaderCompilationError(Exception):
    """Exception raised when shader compilation fails."""
    pass


class ShaderManager:
    """Manages OpenGL shader programs and their compilation."""
    
    def __init__(self):
        self._programs: Dict[str, int] = {}
        self._current_program: Optional[int] = None
    
    def load_compute_shader(self, filepath: str, program_name: str = None) -> int:
        """
        Load and compile a compute shader from file.
        
        Args:
            filepath: Path to the compute shader file
            program_name: Optional name to store the program (defaults to filename)
            
        Returns:
            OpenGL program ID
            
        Raises:
            ShaderCompilationError: If compilation fails
            FileNotFoundError: If shader file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Shader file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Create and compile compute shader
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        
        # Check compilation status
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(shader).decode()
            gl.glDeleteShader(shader)
            raise ShaderCompilationError(f"Compute shader compilation failed: {error}")
        
        # Create program and attach shader
        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        
        # Check linking status
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(program).decode()
            gl.glDeleteShader(shader)
            gl.glDeleteProgram(program)
            raise ShaderCompilationError(f"Program linking failed: {error}")
        
        # Clean up shader object (no longer needed after linking)
        gl.glDeleteShader(shader)
        
        # Store program with name
        name = program_name or os.path.basename(filepath)
        self._programs[name] = program
        
        return program
    
    def load_vertex_fragment_shaders(self, vertex_path: str, fragment_path: str, 
                                   program_name: str = None) -> int:
        """
        Load and compile vertex and fragment shaders from files.
        
        Args:
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
            program_name: Optional name to store the program
            
        Returns:
            OpenGL program ID
            
        Raises:
            ShaderCompilationError: If compilation fails
            FileNotFoundError: If shader files don't exist
        """
        if not os.path.exists(vertex_path):
            raise FileNotFoundError(f"Vertex shader file not found: {vertex_path}")
        if not os.path.exists(fragment_path):
            raise FileNotFoundError(f"Fragment shader file not found: {fragment_path}")
        
        # Load shader sources
        with open(vertex_path, 'r') as f:
            vertex_source = f.read()
        with open(fragment_path, 'r') as f:
            fragment_source = f.read()
        
        # Compile vertex shader
        vertex_shader = self._compile_shader(vertex_source, gl.GL_VERTEX_SHADER, "vertex")
        
        # Compile fragment shader
        fragment_shader = self._compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER, "fragment")
        
        # Create program and link shaders
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        
        # Check linking status
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(program).decode()
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            gl.glDeleteProgram(program)
            raise ShaderCompilationError(f"Program linking failed: {error}")
        
        # Clean up shader objects
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        # Store program with name
        name = program_name or f"{os.path.basename(vertex_path)}_{os.path.basename(fragment_path)}"
        self._programs[name] = program
        
        return program
    
    def _compile_shader(self, source: str, shader_type: int, type_name: str) -> int:
        """
        Compile a shader from source code.
        
        Args:
            source: Shader source code
            shader_type: OpenGL shader type constant
            type_name: Human-readable shader type name for error messages
            
        Returns:
            Compiled shader ID
            
        Raises:
            ShaderCompilationError: If compilation fails
        """
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(shader).decode()
            gl.glDeleteShader(shader)
            raise ShaderCompilationError(f"{type_name.capitalize()} shader compilation failed: {error}")
        
        return shader
    
    def use_program(self, program_id: int) -> None:
        """
        Bind a shader program for use.
        
        Args:
            program_id: OpenGL program ID to bind
        """
        if self._current_program != program_id:
            gl.glUseProgram(program_id)
            self._current_program = program_id
    
    def use_program_by_name(self, name: str) -> None:
        """
        Bind a shader program by its stored name.
        
        Args:
            name: Name of the program to bind
            
        Raises:
            KeyError: If program name not found
        """
        if name not in self._programs:
            raise KeyError(f"Shader program '{name}' not found")
        
        self.use_program(self._programs[name])
    
    def set_uniform(self, name: str, value: Any) -> None:
        """
        Set a uniform variable in the currently bound program.
        
        Args:
            name: Uniform variable name
            value: Value to set (int, float, list/tuple for vectors/matrices)
            
        Raises:
            RuntimeError: If no program is currently bound
        """
        if self._current_program is None:
            raise RuntimeError("No shader program is currently bound")
        
        location = gl.glGetUniformLocation(self._current_program, name)
        if location == -1:
            # Uniform not found - could be optimized out, just return silently
            return
        
        # Handle different value types
        if isinstance(value, bool):
            gl.glUniform1i(location, int(value))
        elif isinstance(value, int):
            gl.glUniform1i(location, value)
        elif isinstance(value, float):
            gl.glUniform1f(location, value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 2:
                gl.glUniform2f(location, *value)
            elif len(value) == 3:
                gl.glUniform3f(location, *value)
            elif len(value) == 4:
                gl.glUniform4f(location, *value)
            elif len(value) == 16:  # 4x4 matrix
                gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, value)
            else:
                raise ValueError(f"Unsupported uniform array size: {len(value)}")
        else:
            raise TypeError(f"Unsupported uniform type: {type(value)}")
    
    def get_program_id(self, name: str) -> int:
        """
        Get the OpenGL program ID by name.
        
        Args:
            name: Program name
            
        Returns:
            OpenGL program ID
            
        Raises:
            KeyError: If program name not found
        """
        if name not in self._programs:
            raise KeyError(f"Shader program '{name}' not found")
        return self._programs[name]
    
    def delete_program(self, name: str) -> None:
        """
        Delete a shader program and free its resources.
        
        Args:
            name: Program name to delete
            
        Raises:
            KeyError: If program name not found
        """
        if name not in self._programs:
            raise KeyError(f"Shader program '{name}' not found")
        
        program_id = self._programs[name]
        if self._current_program == program_id:
            self._current_program = None
            gl.glUseProgram(0)
        
        gl.glDeleteProgram(program_id)
        del self._programs[name]
    
    def cleanup(self) -> None:
        """Delete all shader programs and free resources."""
        for program_id in self._programs.values():
            gl.glDeleteProgram(program_id)
        self._programs.clear()
        self._current_program = None
        gl.glUseProgram(0)
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            self.cleanup()
        except:
            # Ignore errors during cleanup (OpenGL context might be gone)
            pass