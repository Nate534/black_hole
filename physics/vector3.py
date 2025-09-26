"""
Vector3 class for 3D mathematical operations in the black hole simulation.

This module provides a comprehensive Vector3 implementation with all necessary
mathematical operations for physics calculations and 3D graphics.
"""

import math
from dataclasses import dataclass
from typing import Union


@dataclass
class Vector3:
    """
    A 3D vector class with mathematical operations.
    
    Provides essential vector operations including magnitude calculation,
    normalization, dot product, cross product, and basic arithmetic.
    """
    x: float
    y: float
    z: float
    
    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.
        
        Returns:
            float: The magnitude of the vector
        """
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def magnitude_squared(self) -> float:
        """
        Calculate the squared magnitude of the vector.
        
        More efficient than magnitude() when only comparing lengths.
        
        Returns:
            float: The squared magnitude of the vector
        """
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def normalize(self) -> 'Vector3':
        """
        Return a normalized (unit) vector in the same direction.
        
        Returns:
            Vector3: A new normalized vector
            
        Raises:
            ValueError: If the vector has zero magnitude
        """
        mag = self.magnitude()
        if mag == 0.0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3') -> float:
        """
        Calculate the dot product with another vector.
        
        Args:
            other: The other vector
            
        Returns:
            float: The dot product
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        """
        Calculate the cross product with another vector.
        
        Args:
            other: The other vector
            
        Returns:
            Vector3: The cross product vector
        """
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def distance_to(self, other: 'Vector3') -> float:
        """
        Calculate the distance to another vector.
        
        Args:
            other: The other vector
            
        Returns:
            float: The distance between the vectors
        """
        return (self - other).magnitude()
    
    def distance_squared_to(self, other: 'Vector3') -> float:
        """
        Calculate the squared distance to another vector.
        
        More efficient than distance_to() when only comparing distances.
        
        Args:
            other: The other vector
            
        Returns:
            float: The squared distance between the vectors
        """
        return (self - other).magnitude_squared()
    
    # Arithmetic operations
    def __add__(self, other: 'Vector3') -> 'Vector3':
        """Add two vectors."""
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """Subtract two vectors."""
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: Union[float, int]) -> 'Vector3':
        """Multiply vector by a scalar."""
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: Union[float, int]) -> 'Vector3':
        """Multiply vector by a scalar (reverse)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[float, int]) -> 'Vector3':
        """Divide vector by a scalar."""
        if scalar == 0.0:
            raise ValueError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        """Negate the vector."""
        return Vector3(-self.x, -self.y, -self.z)
    
    def __eq__(self, other: 'Vector3') -> bool:
        """Check equality with another vector."""
        return (abs(self.x - other.x) < 1e-9 and 
                abs(self.y - other.y) < 1e-9 and 
                abs(self.z - other.z) < 1e-9)
    
    def __str__(self) -> str:
        """String representation of the vector."""
        return f"Vector3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the vector."""
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"
    
    @classmethod
    def zero(cls) -> 'Vector3':
        """Create a zero vector."""
        return cls(0.0, 0.0, 0.0)
    
    @classmethod
    def unit_x(cls) -> 'Vector3':
        """Create a unit vector along the X axis."""
        return cls(1.0, 0.0, 0.0)
    
    @classmethod
    def unit_y(cls) -> 'Vector3':
        """Create a unit vector along the Y axis."""
        return cls(0.0, 1.0, 0.0)
    
    @classmethod
    def unit_z(cls) -> 'Vector3':
        """Create a unit vector along the Z axis."""
        return cls(0.0, 0.0, 1.0)