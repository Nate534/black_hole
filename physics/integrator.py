"""
PhysicsIntegrator class for numerical integration algorithms in the black hole simulation.

This module implements various integration methods including RK4 (Runge-Kutta 4th order)
for accurate particle trajectory calculations, geodesic computations, and relativistic effects.
"""

import math
from typing import List, Callable, Tuple
from dataclasses import dataclass
from .vector3 import Vector3
from .black_hole import BlackHole
from .particle import Particle


@dataclass
class IntegrationState:
    """
    Represents the state of a particle for integration calculations.
    
    This is used internally by integration algorithms to store
    position and velocity at different integration steps.
    """
    position: Vector3
    velocity: Vector3
    
    def __add__(self, other: 'IntegrationState') -> 'IntegrationState':
        """Add two integration states."""
        return IntegrationState(
            self.position + other.position,
            self.velocity + other.velocity
        )
    
    def __mul__(self, scalar: float) -> 'IntegrationState':
        """Multiply integration state by scalar."""
        return IntegrationState(
            self.position * scalar,
            self.velocity * scalar
        )
    
    def __rmul__(self, scalar: float) -> 'IntegrationState':
        """Multiply integration state by scalar (reverse)."""
        return self.__mul__(scalar)


class PhysicsIntegrator:
    """
    Physics integration algorithms for particle trajectory calculations.
    
    Provides various numerical integration methods including RK4, geodesic calculations,
    and relativistic effects for accurate black hole physics simulation.
    """
    
    def __init__(self):
        """Initialize the physics integrator."""
        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
        self.c = 299792458.0  # Speed of light (m/s)
        self.c_squared = self.c * self.c
        
        # Integration parameters
        self.enable_relativistic_effects = True
        self.enable_geodesic_calculation = True
        self.max_velocity_fraction = 0.1  # Maximum velocity as fraction of c
    
    def integrate_rk4(self, particles: List[Particle], black_hole: BlackHole, dt: float) -> None:
        """
        Integrate particle trajectories using 4th-order Runge-Kutta method.
        
        RK4 provides high accuracy for smooth trajectories by evaluating
        derivatives at multiple points within each timestep.
        
        Args:
            particles: List of particles to integrate
            black_hole: Black hole providing gravitational field
            dt: Time step in seconds
        """
        for particle in particles:
            if not particle.active:
                continue
            
            # Store initial state
            initial_state = IntegrationState(
                Vector3(particle.position.x, particle.position.y, particle.position.z),
                Vector3(particle.velocity.x, particle.velocity.y, particle.velocity.z)
            )
            
            # RK4 integration steps
            k1 = self._calculate_derivative(initial_state, black_hole, particle.mass)
            
            state_k2 = initial_state + k1 * (dt / 2.0)
            k2 = self._calculate_derivative(state_k2, black_hole, particle.mass)
            
            state_k3 = initial_state + k2 * (dt / 2.0)
            k3 = self._calculate_derivative(state_k3, black_hole, particle.mass)
            
            state_k4 = initial_state + k3 * dt
            k4 = self._calculate_derivative(state_k4, black_hole, particle.mass)
            
            # Combine derivatives using RK4 formula
            final_derivative = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0)
            
            # Update particle state
            new_state = initial_state + final_derivative * dt
            
            # Apply relativistic velocity limiting
            if self.enable_relativistic_effects:
                new_state.velocity = self._limit_velocity(new_state.velocity)
            
            # Update particle
            particle.position = new_state.position
            particle.velocity = new_state.velocity
            
            # Apply additional relativistic effects
            if self.enable_relativistic_effects:
                self.apply_relativistic_effects(particle, black_hole)
    
    def _calculate_derivative(self, state: IntegrationState, black_hole: BlackHole, mass: float) -> IntegrationState:
        """
        Calculate the derivative (velocity and acceleration) at a given state.
        
        Args:
            state: Current position and velocity state
            black_hole: Black hole providing gravitational field
            mass: Mass of the particle
            
        Returns:
            IntegrationState: Derivative containing velocity and acceleration
        """
        # Velocity derivative is just the current velocity (dx/dt = v)
        velocity_derivative = state.velocity
        
        # Acceleration derivative is gravitational acceleration
        if self.enable_geodesic_calculation:
            acceleration = self.calculate_geodesic_acceleration(state.position, state.velocity, black_hole)
        else:
            # Simple Newtonian gravity
            gravitational_force = black_hole.calculate_gravitational_force(state.position, mass)
            acceleration = gravitational_force / mass
        
        return IntegrationState(velocity_derivative, acceleration)
    
    def calculate_geodesic_acceleration(self, position: Vector3, velocity: Vector3, black_hole: BlackHole) -> Vector3:
        """
        Calculate geodesic acceleration including relativistic effects.
        
        This implements a simplified version of geodesic motion in Schwarzschild spacetime,
        including corrections for general relativistic effects near the black hole.
        
        Args:
            position: Current position of particle
            velocity: Current velocity of particle
            black_hole: Black hole providing the gravitational field
            
        Returns:
            Vector3: Geodesic acceleration vector
        """
        # Calculate distance and direction to black hole
        displacement = position - black_hole.position
        r = displacement.magnitude()
        
        if r == 0.0:
            return Vector3.zero()  # Avoid singularity
        
        # Schwarzschild radius
        rs = black_hole.get_schwarzschild_radius()
        
        # Newtonian acceleration (base term)
        newtonian_accel = displacement.normalize() * (-self.G * black_hole.mass / (r * r))
        
        if not self.enable_relativistic_effects or r > 10.0 * rs:
            # Far from black hole, use Newtonian physics
            return newtonian_accel
        
        # Relativistic corrections for close approach
        # These are approximations of the full Einstein field equations
        
        # Radial velocity component
        r_hat = displacement.normalize()
        v_radial = velocity.dot(r_hat)
        
        # Tangential velocity component
        v_tangential_vec = velocity - r_hat * v_radial
        v_tangential = v_tangential_vec.magnitude()
        
        # Post-Newtonian corrections (first-order approximations)
        
        # 1. Velocity-dependent correction (1PN term)
        velocity_correction = newtonian_accel * (velocity.magnitude_squared() / self.c_squared)
        
        # 2. Radial velocity correction
        radial_correction = r_hat * (4.0 * self.G * black_hole.mass * v_radial / (r * self.c_squared))
        
        # 3. Frame-dragging-like effect (simplified)
        if v_tangential > 0:
            tangential_correction = v_tangential_vec * (2.0 * self.G * black_hole.mass / (r * self.c_squared))
        else:
            tangential_correction = Vector3.zero()
        
        # 4. Additional relativistic correction for close approach
        proximity_factor = rs / r
        proximity_correction = newtonian_accel * (proximity_factor * proximity_factor)
        
        # Combine all corrections
        total_acceleration = (newtonian_accel + 
                            velocity_correction + 
                            radial_correction + 
                            tangential_correction + 
                            proximity_correction)
        
        return total_acceleration
    
    def apply_relativistic_effects(self, particle: Particle, black_hole: BlackHole) -> None:
        """
        Apply relativistic effects to a particle.
        
        This includes time dilation effects, velocity limiting, and
        other relativistic corrections for particles near the black hole.
        
        Args:
            particle: Particle to apply effects to
            black_hole: Black hole providing gravitational field
        """
        if not particle.active:
            return
        
        # Calculate distance to black hole
        r = particle.position.distance_to(black_hole.position)
        rs = black_hole.get_schwarzschild_radius()
        
        # Apply velocity limiting (cannot exceed speed of light)
        particle.velocity = self._limit_velocity(particle.velocity)
        
        # Time dilation effect (simplified)
        if r > rs:  # Outside event horizon
            # Gravitational time dilation factor
            time_dilation_factor = math.sqrt(1.0 - rs / r)
            
            # Apply time dilation to velocity (particles appear to slow down)
            if time_dilation_factor > 0.1:  # Avoid extreme values
                particle.velocity = particle.velocity * time_dilation_factor
        
        # Additional relativistic mass effect (simplified)
        speed = particle.velocity.magnitude()
        if speed > 0.01 * self.c:  # Only apply for significant velocities
            gamma = 1.0 / math.sqrt(1.0 - (speed * speed) / self.c_squared)
            # This affects how the particle responds to forces (increased inertia)
            # We don't modify mass directly, but this could be used in force calculations
    
    def _limit_velocity(self, velocity: Vector3) -> Vector3:
        """
        Limit velocity to a fraction of the speed of light.
        
        Args:
            velocity: Input velocity vector
            
        Returns:
            Vector3: Limited velocity vector
        """
        speed = velocity.magnitude()
        max_speed = self.max_velocity_fraction * self.c
        
        if speed > max_speed:
            return velocity.normalize() * max_speed
        
        return velocity
    
    def integrate_euler(self, particles: List[Particle], black_hole: BlackHole, dt: float) -> None:
        """
        Integrate particle trajectories using simple Euler method.
        
        This is less accurate than RK4 but faster for testing purposes.
        
        Args:
            particles: List of particles to integrate
            black_hole: Black hole providing gravitational field
            dt: Time step in seconds
        """
        for particle in particles:
            if not particle.active:
                continue
            
            # Calculate acceleration
            if self.enable_geodesic_calculation:
                acceleration = self.calculate_geodesic_acceleration(
                    particle.position, particle.velocity, black_hole
                )
            else:
                gravitational_force = black_hole.calculate_gravitational_force(
                    particle.position, particle.mass
                )
                acceleration = gravitational_force / particle.mass
            
            # Update velocity and position
            particle.velocity = particle.velocity + acceleration * dt
            
            if self.enable_relativistic_effects:
                particle.velocity = self._limit_velocity(particle.velocity)
            
            particle.position = particle.position + particle.velocity * dt
            
            if self.enable_relativistic_effects:
                self.apply_relativistic_effects(particle, black_hole)
    
    def integrate_verlet(self, particles: List[Particle], black_hole: BlackHole, dt: float) -> None:
        """
        Integrate particle trajectories using Velocity Verlet method.
        
        This method is symplectic and conserves energy better than Euler,
        making it good for long-term orbital simulations.
        
        Args:
            particles: List of particles to integrate
            black_hole: Black hole providing gravitational field
            dt: Time step in seconds
        """
        for particle in particles:
            if not particle.active:
                continue
            
            # Calculate current acceleration
            if self.enable_geodesic_calculation:
                current_accel = self.calculate_geodesic_acceleration(
                    particle.position, particle.velocity, black_hole
                )
            else:
                gravitational_force = black_hole.calculate_gravitational_force(
                    particle.position, particle.mass
                )
                current_accel = gravitational_force / particle.mass
            
            # Update position
            particle.position = (particle.position + 
                               particle.velocity * dt + 
                               current_accel * (0.5 * dt * dt))
            
            # Calculate new acceleration at new position
            if self.enable_geodesic_calculation:
                new_accel = self.calculate_geodesic_acceleration(
                    particle.position, particle.velocity, black_hole
                )
            else:
                gravitational_force = black_hole.calculate_gravitational_force(
                    particle.position, particle.mass
                )
                new_accel = gravitational_force / particle.mass
            
            # Update velocity using average acceleration
            particle.velocity = particle.velocity + (current_accel + new_accel) * (0.5 * dt)
            
            if self.enable_relativistic_effects:
                particle.velocity = self._limit_velocity(particle.velocity)
                self.apply_relativistic_effects(particle, black_hole)
    
    def calculate_orbital_parameters(self, particle: Particle, black_hole: BlackHole) -> dict:
        """
        Calculate orbital parameters for a particle around the black hole.
        
        Args:
            particle: Particle to analyze
            black_hole: Central black hole
            
        Returns:
            dict: Dictionary containing orbital parameters
        """
        if not particle.active:
            return {}
        
        # Position and velocity relative to black hole
        r_vec = particle.position - black_hole.position
        v_vec = particle.velocity
        
        r = r_vec.magnitude()
        v = v_vec.magnitude()
        
        # Specific orbital energy
        gravitational_potential = -self.G * black_hole.mass / r
        kinetic_energy_per_mass = 0.5 * v * v
        specific_energy = kinetic_energy_per_mass + gravitational_potential
        
        # Angular momentum per unit mass
        h_vec = r_vec.cross(v_vec)
        specific_angular_momentum = h_vec.magnitude()
        
        # Eccentricity vector and magnitude
        mu = self.G * black_hole.mass
        e_vec = ((v_vec.cross(h_vec)) / mu) - (r_vec / r)
        eccentricity = e_vec.magnitude()
        
        # Semi-major axis (for bound orbits)
        if specific_energy < 0:
            semi_major_axis = -mu / (2.0 * specific_energy)
        else:
            semi_major_axis = float('inf')  # Unbound orbit
        
        # Periapsis and apoapsis distances
        if eccentricity < 1.0 and semi_major_axis > 0:
            periapsis = semi_major_axis * (1.0 - eccentricity)
            apoapsis = semi_major_axis * (1.0 + eccentricity)
        else:
            periapsis = specific_angular_momentum * specific_angular_momentum / (mu * (1.0 + eccentricity))
            apoapsis = float('inf')
        
        return {
            'specific_energy': specific_energy,
            'specific_angular_momentum': specific_angular_momentum,
            'eccentricity': eccentricity,
            'semi_major_axis': semi_major_axis,
            'periapsis': periapsis,
            'apoapsis': apoapsis,
            'is_bound': specific_energy < 0,
            'current_distance': r,
            'current_speed': v
        }
    
    def set_integration_parameters(self, 
                                 enable_relativistic: bool = True,
                                 enable_geodesic: bool = True,
                                 max_velocity_fraction: float = 0.1) -> None:
        """
        Set integration parameters.
        
        Args:
            enable_relativistic: Whether to apply relativistic effects
            enable_geodesic: Whether to use geodesic calculations
            max_velocity_fraction: Maximum velocity as fraction of speed of light
        """
        self.enable_relativistic_effects = enable_relativistic
        self.enable_geodesic_calculation = enable_geodesic
        self.max_velocity_fraction = max(0.01, min(0.99, max_velocity_fraction))
    
    def __str__(self) -> str:
        """String representation of the integrator."""
        return (f"PhysicsIntegrator(relativistic={self.enable_relativistic_effects}, "
                f"geodesic={self.enable_geodesic_calculation}, "
                f"max_v={self.max_velocity_fraction:.2f}c)")