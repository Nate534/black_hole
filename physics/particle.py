"""
Particle class for dynamic objects in the black hole simulation.

This module implements the Particle class with position, velocity, mass properties
and methods for position updates and force application.
"""

from dataclasses import dataclass
from typing import List
from .vector3 import Vector3


@dataclass
class Particle:
    """
    Represents a particle with position, velocity, and mass.
    
    This class encapsulates particle dynamics including position updates,
    force application, energy calculations, and state management.
    """
    mass: float  # Mass in kg
    position: Vector3  # Position in 3D space (m)
    velocity: Vector3  # Velocity vector (m/s)
    active: bool = True  # Whether particle is active in simulation
    
    def __post_init__(self):
        """Initialize additional particle properties after dataclass creation."""
        self._force_accumulator = Vector3.zero()  # Accumulated forces for this timestep
        self._trail_positions: List[Vector3] = []  # Position history for trails
        self._max_trail_length = 100  # Maximum trail length
    
    def update_position(self, dt: float) -> None:
        """
        Update particle position based on current velocity.
        
        Uses simple Euler integration: position += velocity * dt
        
        Args:
            dt: Time step in seconds
        """
        if not self.active:
            return
            
        # Update position using current velocity
        self.position = self.position + (self.velocity * dt)
        
        # Add current position to trail
        self._add_to_trail(self.position)
    
    def apply_force(self, force: Vector3, dt: float) -> None:
        """
        Apply a force to the particle for the current timestep.
        
        Forces are accumulated and will be used to update velocity
        when integrate_forces() is called.
        
        Args:
            force: Force vector in Newtons
            dt: Time step in seconds (not used in accumulation, but kept for interface consistency)
        """
        if not self.active:
            return
            
        self._force_accumulator = self._force_accumulator + force
    
    def integrate_forces(self, dt: float) -> None:
        """
        Integrate accumulated forces to update velocity.
        
        Uses Newton's second law: F = ma, so a = F/m
        Then updates velocity: velocity += acceleration * dt
        
        Args:
            dt: Time step in seconds
        """
        if not self.active or self.mass <= 0:
            return
        
        # Calculate acceleration from accumulated forces
        acceleration = self._force_accumulator / self.mass
        
        # Update velocity
        self.velocity = self.velocity + (acceleration * dt)
        
        # Clear force accumulator for next timestep
        self._force_accumulator = Vector3.zero()
    
    def get_kinetic_energy(self) -> float:
        """
        Calculate the kinetic energy of the particle.
        
        KE = (1/2) * m * v²
        
        Returns:
            float: Kinetic energy in Joules
        """
        if not self.active:
            return 0.0
            
        speed_squared = self.velocity.magnitude_squared()
        return 0.5 * self.mass * speed_squared
    
    def get_momentum(self) -> Vector3:
        """
        Calculate the momentum vector of the particle.
        
        p = m * v
        
        Returns:
            Vector3: Momentum vector in kg⋅m/s
        """
        if not self.active:
            return Vector3.zero()
            
        return self.velocity * self.mass
    
    def get_speed(self) -> float:
        """
        Get the speed (magnitude of velocity) of the particle.
        
        Returns:
            float: Speed in m/s
        """
        if not self.active:
            return 0.0
            
        return self.velocity.magnitude()
    
    def set_active(self, active: bool) -> None:
        """
        Set the active state of the particle.
        
        Inactive particles don't participate in physics updates.
        
        Args:
            active: Whether the particle should be active
        """
        self.active = active
        if not active:
            # Clear forces when deactivating
            self._force_accumulator = Vector3.zero()
    
    def reset_forces(self) -> None:
        """
        Reset accumulated forces to zero.
        
        Useful for manual force management or error recovery.
        """
        self._force_accumulator = Vector3.zero()
    
    def get_accumulated_force(self) -> Vector3:
        """
        Get the currently accumulated force.
        
        Returns:
            Vector3: Total accumulated force in Newtons
        """
        return self._force_accumulator
    
    def _add_to_trail(self, position: Vector3) -> None:
        """
        Add a position to the particle's trail history.
        
        Args:
            position: Position to add to trail
        """
        self._trail_positions.append(Vector3(position.x, position.y, position.z))
        
        # Limit trail length
        if len(self._trail_positions) > self._max_trail_length:
            self._trail_positions.pop(0)
    
    def get_trail_positions(self) -> List[Vector3]:
        """
        Get the particle's position trail.
        
        Returns:
            List[Vector3]: List of historical positions
        """
        return self._trail_positions.copy()
    
    def set_max_trail_length(self, length: int) -> None:
        """
        Set the maximum trail length.
        
        Args:
            length: Maximum number of positions to keep in trail
        """
        self._max_trail_length = max(0, length)
        
        # Trim existing trail if necessary
        while len(self._trail_positions) > self._max_trail_length:
            self._trail_positions.pop(0)
    
    def clear_trail(self) -> None:
        """Clear the particle's position trail."""
        self._trail_positions.clear()
    
    def distance_to(self, other_position: Vector3) -> float:
        """
        Calculate distance to another position.
        
        Args:
            other_position: Position to calculate distance to
            
        Returns:
            float: Distance in meters
        """
        return self.position.distance_to(other_position)
    
    def distance_to_particle(self, other_particle: 'Particle') -> float:
        """
        Calculate distance to another particle.
        
        Args:
            other_particle: Other particle to calculate distance to
            
        Returns:
            float: Distance in meters
        """
        return self.position.distance_to(other_particle.position)
    
    def copy(self) -> 'Particle':
        """
        Create a copy of this particle.
        
        Returns:
            Particle: New particle with same properties
        """
        new_particle = Particle(
            mass=self.mass,
            position=Vector3(self.position.x, self.position.y, self.position.z),
            velocity=Vector3(self.velocity.x, self.velocity.y, self.velocity.z),
            active=self.active
        )
        
        # Copy trail and settings
        new_particle._trail_positions = [Vector3(pos.x, pos.y, pos.z) for pos in self._trail_positions]
        new_particle._max_trail_length = self._max_trail_length
        new_particle._force_accumulator = Vector3(
            self._force_accumulator.x, 
            self._force_accumulator.y, 
            self._force_accumulator.z
        )
        
        return new_particle
    
    def __str__(self) -> str:
        """String representation of the particle."""
        status = "active" if self.active else "inactive"
        return f"Particle(mass={self.mass:.2e} kg, pos={self.position}, vel={self.velocity}, {status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the particle."""
        ke = self.get_kinetic_energy()
        speed = self.get_speed()
        return f"Particle(mass={self.mass}, position={self.position}, velocity={self.velocity}, " \
               f"active={self.active}, KE={ke:.2e}J, speed={speed:.2e}m/s)"


class ParticleSystem:
    """
    Manages a collection of particles for efficient batch operations.
    
    Provides methods for updating multiple particles, applying forces,
    and managing particle lifecycle.
    """
    
    def __init__(self):
        """Initialize empty particle system."""
        self.particles: List[Particle] = []
    
    def add_particle(self, particle: Particle) -> None:
        """
        Add a particle to the system.
        
        Args:
            particle: Particle to add
        """
        self.particles.append(particle)
    
    def remove_particle(self, particle: Particle) -> bool:
        """
        Remove a particle from the system.
        
        Args:
            particle: Particle to remove
            
        Returns:
            bool: True if particle was found and removed
        """
        try:
            self.particles.remove(particle)
            return True
        except ValueError:
            return False
    
    def get_active_particles(self) -> List[Particle]:
        """
        Get all active particles.
        
        Returns:
            List[Particle]: List of active particles
        """
        return [p for p in self.particles if p.active]
    
    def update_all_positions(self, dt: float) -> None:
        """
        Update positions of all active particles.
        
        Args:
            dt: Time step in seconds
        """
        for particle in self.particles:
            if particle.active:
                particle.update_position(dt)
    
    def integrate_all_forces(self, dt: float) -> None:
        """
        Integrate forces for all active particles.
        
        Args:
            dt: Time step in seconds
        """
        for particle in self.particles:
            if particle.active:
                particle.integrate_forces(dt)
    
    def reset_all_forces(self) -> None:
        """Reset forces for all particles."""
        for particle in self.particles:
            particle.reset_forces()
    
    def get_total_kinetic_energy(self) -> float:
        """
        Calculate total kinetic energy of all active particles.
        
        Returns:
            float: Total kinetic energy in Joules
        """
        return sum(p.get_kinetic_energy() for p in self.particles if p.active)
    
    def get_center_of_mass(self) -> Vector3:
        """
        Calculate center of mass of all active particles.
        
        Returns:
            Vector3: Center of mass position
        """
        active_particles = self.get_active_particles()
        if not active_particles:
            return Vector3.zero()
        
        total_mass = sum(p.mass for p in active_particles)
        if total_mass == 0:
            return Vector3.zero()
        
        weighted_position = Vector3.zero()
        for particle in active_particles:
            weighted_position = weighted_position + (particle.position * particle.mass)
        
        return weighted_position / total_mass
    
    def clear(self) -> None:
        """Remove all particles from the system."""
        self.particles.clear()
    
    def __len__(self) -> int:
        """Get number of particles in system."""
        return len(self.particles)
    
    def __iter__(self):
        """Iterate over particles."""
        return iter(self.particles)
    
    def __str__(self) -> str:
        """String representation of particle system."""
        active_count = len(self.get_active_particles())
        return f"ParticleSystem({len(self.particles)} particles, {active_count} active)"