"""
Unit tests for the BlackHole class.

Tests gravitational calculations, Schwarzschild radius computation,
event horizon detection, and other black hole physics.
"""

import math
import pytest
from physics.vector3 import Vector3
from physics.black_hole import BlackHole


class TestBlackHole:
    """Test cases for BlackHole class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Solar mass black hole at origin
        self.solar_mass = 1.989e30  # kg
        self.black_hole = BlackHole(
            mass=self.solar_mass,
            position=Vector3.zero()
        )
        
        # Test particle
        self.particle_mass = 1000.0  # kg
        self.test_position = Vector3(1000.0, 0.0, 0.0)  # 1 km from center
    
    def test_schwarzschild_radius_calculation(self):
        """Test Schwarzschild radius calculation for known values."""
        # For solar mass: rs = 2GM/c² ≈ 2953 meters
        expected_rs = (2.0 * BlackHole.G * self.solar_mass) / (BlackHole.c ** 2)
        calculated_rs = self.black_hole.get_schwarzschild_radius()
        
        assert abs(calculated_rs - expected_rs) < 1e-6
        assert calculated_rs > 0
        
        # Test with different mass
        small_black_hole = BlackHole(mass=1e20, position=Vector3.zero())
        small_rs = small_black_hole.get_schwarzschild_radius()
        assert small_rs < calculated_rs  # Smaller mass = smaller radius
    
    def test_photon_sphere_radius(self):
        """Test photon sphere radius calculation."""
        rs = self.black_hole.get_schwarzschild_radius()
        photon_sphere = self.black_hole.get_photon_sphere_radius()
        
        # Photon sphere should be 1.5 times Schwarzschild radius
        assert abs(photon_sphere - 1.5 * rs) < 1e-6
        assert photon_sphere > rs
    
    def test_gravitational_force_calculation(self):
        """Test gravitational force calculation."""
        force = self.black_hole.calculate_gravitational_force(
            self.test_position, self.particle_mass
        )
        
        # Force should point toward black hole (negative x direction)
        assert force.x < 0
        assert abs(force.y) < 1e-10  # Should be zero
        assert abs(force.z) < 1e-10  # Should be zero
        
        # Calculate expected force magnitude: F = GMm/r²
        distance = self.test_position.magnitude()
        expected_magnitude = (BlackHole.G * self.solar_mass * self.particle_mass) / (distance ** 2)
        calculated_magnitude = force.magnitude()
        
        assert abs(calculated_magnitude - expected_magnitude) < 1e-6
    
    def test_gravitational_force_direction(self):
        """Test that gravitational force always points toward black hole."""
        positions = [
            Vector3(100.0, 0.0, 0.0),
            Vector3(0.0, 200.0, 0.0),
            Vector3(0.0, 0.0, 300.0),
            Vector3(100.0, 100.0, 100.0)
        ]
        
        for pos in positions:
            force = self.black_hole.calculate_gravitational_force(pos, self.particle_mass)
            
            # Force should point from particle toward black hole
            to_black_hole = self.black_hole.position - pos
            force_direction = force.normalize()
            expected_direction = to_black_hole.normalize()
            
            # Directions should be the same (dot product ≈ 1)
            dot_product = force_direction.dot(expected_direction)
            assert abs(dot_product - 1.0) < 1e-6
    
    def test_gravitational_force_zero_distance_error(self):
        """Test that gravitational force raises error at zero distance."""
        with pytest.raises(ValueError, match="same position"):
            self.black_hole.calculate_gravitational_force(
                self.black_hole.position, self.particle_mass
            )
    
    def test_event_horizon_detection(self):
        """Test event horizon detection."""
        rs = self.black_hole.get_schwarzschild_radius()
        
        # Position inside event horizon
        inside_pos = Vector3(rs * 0.5, 0.0, 0.0)
        assert self.black_hole.is_within_event_horizon(inside_pos)
        
        # Position on event horizon
        on_horizon_pos = Vector3(rs, 0.0, 0.0)
        assert self.black_hole.is_within_event_horizon(on_horizon_pos)
        
        # Position outside event horizon
        outside_pos = Vector3(rs * 2.0, 0.0, 0.0)
        assert not self.black_hole.is_within_event_horizon(outside_pos)
    
    def test_photon_sphere_detection(self):
        """Test photon sphere detection."""
        photon_radius = self.black_hole.get_photon_sphere_radius()
        
        # Position inside photon sphere
        inside_pos = Vector3(photon_radius * 0.5, 0.0, 0.0)
        assert self.black_hole.is_within_photon_sphere(inside_pos)
        
        # Position on photon sphere
        on_sphere_pos = Vector3(photon_radius, 0.0, 0.0)
        assert self.black_hole.is_within_photon_sphere(on_sphere_pos)
        
        # Position outside photon sphere
        outside_pos = Vector3(photon_radius * 2.0, 0.0, 0.0)
        assert not self.black_hole.is_within_photon_sphere(outside_pos)
    
    def test_spacetime_curvature(self):
        """Test spacetime curvature calculation."""
        rs = self.black_hole.get_schwarzschild_radius()
        
        # At event horizon, curvature should be maximum (1.0)
        horizon_pos = Vector3(rs, 0.0, 0.0)
        curvature_at_horizon = self.black_hole.get_spacetime_curvature(horizon_pos)
        assert abs(curvature_at_horizon - 1.0) < 1e-6
        
        # Inside event horizon, curvature should be 1.0
        inside_pos = Vector3(rs * 0.5, 0.0, 0.0)
        curvature_inside = self.black_hole.get_spacetime_curvature(inside_pos)
        assert abs(curvature_inside - 1.0) < 1e-6
        
        # Far away, curvature should be small
        far_pos = Vector3(rs * 100.0, 0.0, 0.0)
        curvature_far = self.black_hole.get_spacetime_curvature(far_pos)
        assert curvature_far < 0.01
        assert curvature_far >= 0.0
        
        # Curvature should decrease with distance
        mid_pos = Vector3(rs * 10.0, 0.0, 0.0)
        curvature_mid = self.black_hole.get_spacetime_curvature(mid_pos)
        assert curvature_mid > curvature_far
        assert curvature_mid < curvature_at_horizon
    
    def test_escape_velocity(self):
        """Test escape velocity calculation."""
        distance = 1000.0  # meters
        position = Vector3(distance, 0.0, 0.0)
        
        escape_vel = self.black_hole.get_escape_velocity(position)
        
        # Calculate expected: v = sqrt(2GM/r)
        expected_vel = math.sqrt((2.0 * BlackHole.G * self.solar_mass) / distance)
        
        assert abs(escape_vel - expected_vel) < 1e-6
        assert escape_vel > 0
    
    def test_escape_velocity_zero_distance_error(self):
        """Test that escape velocity raises error at zero distance."""
        with pytest.raises(ValueError, match="black hole center"):
            self.black_hole.get_escape_velocity(self.black_hole.position)
    
    def test_orbital_velocity(self):
        """Test orbital velocity calculation."""
        distance = 1000.0  # meters
        position = Vector3(distance, 0.0, 0.0)
        
        orbital_vel = self.black_hole.get_orbital_velocity(position)
        
        # Calculate expected: v = sqrt(GM/r)
        expected_vel = math.sqrt((BlackHole.G * self.solar_mass) / distance)
        
        assert abs(orbital_vel - expected_vel) < 1e-6
        assert orbital_vel > 0
        
        # Orbital velocity should be less than escape velocity
        escape_vel = self.black_hole.get_escape_velocity(position)
        assert orbital_vel < escape_vel
    
    def test_orbital_velocity_zero_distance_error(self):
        """Test that orbital velocity raises error at zero distance."""
        with pytest.raises(ValueError, match="black hole center"):
            self.black_hole.get_orbital_velocity(self.black_hole.position)
    
    def test_force_inverse_square_law(self):
        """Test that gravitational force follows inverse square law."""
        # Test at different distances
        distances = [1000.0, 2000.0, 4000.0]
        forces = []
        
        for distance in distances:
            pos = Vector3(distance, 0.0, 0.0)
            force = self.black_hole.calculate_gravitational_force(pos, self.particle_mass)
            forces.append(force.magnitude())
        
        # Force should be inversely proportional to distance squared
        # F₁/F₂ = (r₂/r₁)²
        ratio_12 = forces[0] / forces[1]
        expected_ratio_12 = (distances[1] / distances[0]) ** 2
        assert abs(ratio_12 - expected_ratio_12) < 1e-6
        
        ratio_23 = forces[1] / forces[2]
        expected_ratio_23 = (distances[2] / distances[1]) ** 2
        assert abs(ratio_23 - expected_ratio_23) < 1e-6
    
    def test_black_hole_with_offset_position(self):
        """Test black hole calculations with non-zero position."""
        offset_position = Vector3(100.0, 200.0, 300.0)
        offset_black_hole = BlackHole(mass=self.solar_mass, position=offset_position)
        
        # Test particle position
        particle_pos = Vector3(200.0, 200.0, 300.0)  # 100m away in x direction
        
        force = offset_black_hole.calculate_gravitational_force(particle_pos, self.particle_mass)
        
        # Force should point toward black hole
        to_black_hole = offset_position - particle_pos
        force_direction = force.normalize()
        expected_direction = to_black_hole.normalize()
        
        dot_product = force_direction.dot(expected_direction)
        assert abs(dot_product - 1.0) < 1e-6
    
    def test_string_representations(self):
        """Test string representations of BlackHole."""
        str_repr = str(self.black_hole)
        assert "BlackHole" in str_repr
        assert str(self.solar_mass) in str_repr or f"{self.solar_mass:.2e}" in str_repr
        
        repr_str = repr(self.black_hole)
        assert "BlackHole" in repr_str
        assert "rs=" in repr_str  # Should include Schwarzschild radius