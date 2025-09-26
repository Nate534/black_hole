"""
BlackHole class for gravitational calculations in the black hole simulation.

This module implements the BlackHole class with methods for calculating
gravitational forces, Schwarzschild radius, event horizon detection,
and other relativistic effects.
"""

import math
from dataclasses import dataclass
from .vector3 import Vector3


@dataclass
class BlackHole:
    """
    Represents a black hole with gravitational effects.
    
    This class encapsulates black hole physics including mass, position,
    Schwarzschild radius calculation, gravitational force computation,
    and event horizon detection.
    """
    mass: float  # Mass in kg
    position: Vector3  # Position in 3D space
    
    # Physical constants
    G: float = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
    c: float = 299792458.0  # Speed of light (m/s)
    
    def get_schwarzschild_radius(self) -> float:
        """
        Calculate the Schwarzschild radius (event horizon radius).
        
        The Schwarzschild radius is given by: rs = 2GM/c²
        
        Returns:
            float: The Schwarzschild radius in meters
        """
        return (2.0 * self.G * self.mass) / (self.c * self.c)
    
    def get_photon_sphere_radius(self) -> float:
        """
        Calculate the photon sphere radius.
        
        The photon sphere is at 1.5 times the Schwarzschild radius,
        where photons can orbit the black hole in unstable circular orbits.
        
        Returns:
            float: The photon sphere radius in meters
        """
        return 1.5 * self.get_schwarzschild_radius()
    
    def calculate_gravitational_force(self, particle_pos: Vector3, particle_mass: float) -> Vector3:
        """
        Calculate the gravitational force on a particle.
        
        Uses Newton's law of universal gravitation: F = GMm/r²
        The force direction points toward the black hole.
        
        Args:
            particle_pos: Position of the particle
            particle_mass: Mass of the particle in kg
            
        Returns:
            Vector3: Gravitational force vector in Newtons
            
        Raises:
            ValueError: If particle is at the same position as black hole
        """
        # Calculate displacement vector from black hole to particle
        displacement = particle_pos - self.position
        distance = displacement.magnitude()
        
        if distance == 0.0:
            raise ValueError("Particle cannot be at the same position as black hole")
        
        # Calculate force magnitude: F = GMm/r²
        force_magnitude = (self.G * self.mass * particle_mass) / (distance * distance)
        
        # Force direction is toward the black hole (negative displacement)
        force_direction = displacement.normalize() * -1.0
        
        return force_direction * force_magnitude
    
    def is_within_event_horizon(self, particle_pos: Vector3) -> bool:
        """
        Check if a particle is within the event horizon.
        
        Args:
            particle_pos: Position of the particle
            
        Returns:
            bool: True if particle is within the event horizon
        """
        distance = self.position.distance_to(particle_pos)
        return distance <= self.get_schwarzschild_radius()
    
    def is_within_photon_sphere(self, particle_pos: Vector3) -> bool:
        """
        Check if a particle is within the photon sphere.
        
        Args:
            particle_pos: Position of the particle
            
        Returns:
            bool: True if particle is within the photon sphere
        """
        distance = self.position.distance_to(particle_pos)
        return distance <= self.get_photon_sphere_radius()
    
    def get_spacetime_curvature(self, position: Vector3) -> float:
        """
        Calculate a simplified measure of spacetime curvature at a position.
        
        This provides a normalized measure of gravitational field strength
        that can be used for visual effects or physics approximations.
        
        Args:
            position: Position to calculate curvature at
            
        Returns:
            float: Normalized curvature value (0.0 to 1.0)
        """
        distance = self.position.distance_to(position)
        schwarzschild_radius = self.get_schwarzschild_radius()
        
        if distance <= schwarzschild_radius:
            return 1.0  # Maximum curvature at event horizon
        
        # Curvature falls off as 1/r² but normalized
        curvature = (schwarzschild_radius / distance) ** 2
        return min(curvature, 1.0)
    
    def get_escape_velocity(self, position: Vector3) -> float:
        """
        Calculate the escape velocity at a given position.
        
        The escape velocity is given by: v = sqrt(2GM/r)
        
        Args:
            position: Position to calculate escape velocity at
            
        Returns:
            float: Escape velocity in m/s
            
        Raises:
            ValueError: If position is at the black hole center
        """
        distance = self.position.distance_to(position)
        
        if distance == 0.0:
            raise ValueError("Cannot calculate escape velocity at black hole center")
        
        return math.sqrt((2.0 * self.G * self.mass) / distance)
    
    def get_orbital_velocity(self, position: Vector3) -> float:
        """
        Calculate the circular orbital velocity at a given position.
        
        The orbital velocity is given by: v = sqrt(GM/r)
        
        Args:
            position: Position to calculate orbital velocity at
            
        Returns:
            float: Orbital velocity in m/s
            
        Raises:
            ValueError: If position is at the black hole center
        """
        distance = self.position.distance_to(position)
        
        if distance == 0.0:
            raise ValueError("Cannot calculate orbital velocity at black hole center")
        
        return math.sqrt((self.G * self.mass) / distance)
    
    def __str__(self) -> str:
        """String representation of the black hole."""
        return f"BlackHole(mass={self.mass:.2e} kg, position={self.position})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the black hole."""
        rs = self.get_schwarzschild_radius()
        return f"BlackHole(mass={self.mass}, position={self.position}, rs={rs:.2e}m)"