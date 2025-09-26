"""
Unit tests for the Particle class and ParticleSystem.

Tests particle dynamics, force application, position updates,
and particle system management.
"""

import pytest
from physics.vector3 import Vector3
from physics.particle import Particle, ParticleSystem


class TestParticle:
    """Test cases for Particle class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.particle = Particle(
            mass=1000.0,  # kg
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3(10.0, 0.0, 0.0)  # m/s
        )
    
    def test_particle_initialization(self):
        """Test particle initialization with default values."""
        assert self.particle.mass == 1000.0
        assert self.particle.position == Vector3.zero()
        assert self.particle.velocity == Vector3(10.0, 0.0, 0.0)
        assert self.particle.active is True
        
        # Test internal state
        assert self.particle._force_accumulator == Vector3.zero()
        assert len(self.particle._trail_positions) == 0
        assert self.particle._max_trail_length == 100
    
    def test_position_update(self):
        """Test position update based on velocity."""
        dt = 0.1  # seconds
        initial_position = Vector3(self.particle.position.x, self.particle.position.y, self.particle.position.z)
        
        self.particle.update_position(dt)
        
        # Position should be updated by velocity * dt
        expected_position = initial_position + (self.particle.velocity * dt)
        assert self.particle.position == expected_position
        
        # Trail should have one entry
        trail = self.particle.get_trail_positions()
        assert len(trail) == 1
        assert trail[0] == expected_position
    
    def test_position_update_inactive_particle(self):
        """Test that inactive particles don't update position."""
        self.particle.set_active(False)
        initial_position = Vector3(self.particle.position.x, self.particle.position.y, self.particle.position.z)
        
        self.particle.update_position(0.1)
        
        assert self.particle.position == initial_position
    
    def test_force_application(self):
        """Test force application and accumulation."""
        force1 = Vector3(100.0, 0.0, 0.0)  # N
        force2 = Vector3(0.0, 50.0, 0.0)   # N
        dt = 0.1
        
        self.particle.apply_force(force1, dt)
        self.particle.apply_force(force2, dt)
        
        # Forces should be accumulated
        accumulated = self.particle.get_accumulated_force()
        expected_total = force1 + force2
        assert accumulated == expected_total
    
    def test_force_integration(self):
        """Test force integration to update velocity."""
        force = Vector3(1000.0, 0.0, 0.0)  # N
        dt = 0.1  # s
        initial_velocity = Vector3(self.particle.velocity.x, self.particle.velocity.y, self.particle.velocity.z)
        
        self.particle.apply_force(force, dt)
        self.particle.integrate_forces(dt)
        
        # Calculate expected velocity change
        # a = F/m, Δv = a * dt
        acceleration = force / self.particle.mass
        expected_velocity = initial_velocity + (acceleration * dt)
        
        assert self.particle.velocity == expected_velocity
        
        # Force accumulator should be reset
        assert self.particle.get_accumulated_force() == Vector3.zero()
    
    def test_force_integration_inactive_particle(self):
        """Test that inactive particles don't integrate forces."""
        self.particle.set_active(False)
        initial_velocity = Vector3(self.particle.velocity.x, self.particle.velocity.y, self.particle.velocity.z)
        
        force = Vector3(1000.0, 0.0, 0.0)
        self.particle.apply_force(force, 0.1)
        self.particle.integrate_forces(0.1)
        
        assert self.particle.velocity == initial_velocity
    
    def test_force_integration_zero_mass(self):
        """Test force integration with zero mass particle."""
        zero_mass_particle = Particle(
            mass=0.0,
            position=Vector3.zero(),
            velocity=Vector3.zero()
        )
        
        initial_velocity = Vector3(zero_mass_particle.velocity.x, zero_mass_particle.velocity.y, zero_mass_particle.velocity.z)
        
        force = Vector3(1000.0, 0.0, 0.0)
        zero_mass_particle.apply_force(force, 0.1)
        zero_mass_particle.integrate_forces(0.1)
        
        # Velocity should not change for zero mass
        assert zero_mass_particle.velocity == initial_velocity
    
    def test_kinetic_energy_calculation(self):
        """Test kinetic energy calculation."""
        # KE = (1/2) * m * v²
        speed_squared = self.particle.velocity.magnitude_squared()
        expected_ke = 0.5 * self.particle.mass * speed_squared
        
        calculated_ke = self.particle.get_kinetic_energy()
        assert abs(calculated_ke - expected_ke) < 1e-6
    
    def test_kinetic_energy_inactive_particle(self):
        """Test that inactive particles have zero kinetic energy."""
        self.particle.set_active(False)
        assert self.particle.get_kinetic_energy() == 0.0
    
    def test_momentum_calculation(self):
        """Test momentum calculation."""
        # p = m * v
        expected_momentum = self.particle.velocity * self.particle.mass
        calculated_momentum = self.particle.get_momentum()
        
        assert calculated_momentum == expected_momentum
    
    def test_momentum_inactive_particle(self):
        """Test that inactive particles have zero momentum."""
        self.particle.set_active(False)
        assert self.particle.get_momentum() == Vector3.zero()
    
    def test_speed_calculation(self):
        """Test speed calculation."""
        expected_speed = self.particle.velocity.magnitude()
        calculated_speed = self.particle.get_speed()
        
        assert abs(calculated_speed - expected_speed) < 1e-6
    
    def test_speed_inactive_particle(self):
        """Test that inactive particles have zero speed."""
        self.particle.set_active(False)
        assert self.particle.get_speed() == 0.0
    
    def test_active_state_management(self):
        """Test particle active state management."""
        assert self.particle.active is True
        
        self.particle.set_active(False)
        assert self.particle.active is False
        
        # Forces should be cleared when deactivating
        self.particle.apply_force(Vector3(100.0, 0.0, 0.0), 0.1)
        self.particle.set_active(False)
        assert self.particle.get_accumulated_force() == Vector3.zero()
    
    def test_force_reset(self):
        """Test manual force reset."""
        force = Vector3(100.0, 50.0, 25.0)
        self.particle.apply_force(force, 0.1)
        
        assert self.particle.get_accumulated_force() == force
        
        self.particle.reset_forces()
        assert self.particle.get_accumulated_force() == Vector3.zero()
    
    def test_trail_management(self):
        """Test particle trail position tracking."""
        positions = [
            Vector3(1.0, 0.0, 0.0),
            Vector3(2.0, 0.0, 0.0),
            Vector3(3.0, 0.0, 0.0)
        ]
        
        for pos in positions:
            self.particle.position = pos
            self.particle.update_position(0.1)  # This adds to trail
        
        trail = self.particle.get_trail_positions()
        assert len(trail) == 3
        
        # Check that positions were added correctly
        for i, pos in enumerate(positions):
            expected_pos = pos + (self.particle.velocity * 0.1)
            assert trail[i] == expected_pos
    
    def test_trail_length_limit(self):
        """Test trail length limiting."""
        self.particle.set_max_trail_length(3)
        
        # Add more positions than the limit
        for i in range(5):
            self.particle.position = Vector3(float(i), 0.0, 0.0)
            self.particle.update_position(0.1)
        
        trail = self.particle.get_trail_positions()
        assert len(trail) == 3  # Should be limited to 3
    
    def test_trail_clear(self):
        """Test trail clearing."""
        # Add some positions
        for i in range(3):
            self.particle.position = Vector3(float(i), 0.0, 0.0)
            self.particle.update_position(0.1)
        
        assert len(self.particle.get_trail_positions()) == 3
        
        self.particle.clear_trail()
        assert len(self.particle.get_trail_positions()) == 0
    
    def test_distance_calculations(self):
        """Test distance calculation methods."""
        other_position = Vector3(3.0, 4.0, 0.0)
        expected_distance = 5.0  # 3-4-5 triangle
        
        calculated_distance = self.particle.distance_to(other_position)
        assert abs(calculated_distance - expected_distance) < 1e-6
        
        # Test distance to another particle
        other_particle = Particle(
            mass=500.0,
            position=other_position,
            velocity=Vector3.zero()
        )
        
        particle_distance = self.particle.distance_to_particle(other_particle)
        assert abs(particle_distance - expected_distance) < 1e-6
    
    def test_particle_copy(self):
        """Test particle copying."""
        # Set up particle with some state
        self.particle.apply_force(Vector3(100.0, 0.0, 0.0), 0.1)
        self.particle.update_position(0.1)  # Add to trail
        self.particle.set_max_trail_length(50)
        
        # Create copy
        copied_particle = self.particle.copy()
        
        # Check that all properties are copied
        assert copied_particle.mass == self.particle.mass
        assert copied_particle.position == self.particle.position
        assert copied_particle.velocity == self.particle.velocity
        assert copied_particle.active == self.particle.active
        assert copied_particle._max_trail_length == self.particle._max_trail_length
        assert copied_particle.get_accumulated_force() == self.particle.get_accumulated_force()
        assert len(copied_particle.get_trail_positions()) == len(self.particle.get_trail_positions())
        
        # Ensure they are separate objects
        assert copied_particle is not self.particle
        assert copied_particle.position is not self.particle.position
    
    def test_string_representations(self):
        """Test string representations of Particle."""
        str_repr = str(self.particle)
        assert "Particle" in str_repr
        assert "active" in str_repr
        assert str(self.particle.mass) in str_repr or f"{self.particle.mass:.2e}" in str_repr
        
        repr_str = repr(self.particle)
        assert "Particle" in repr_str
        assert "KE=" in repr_str  # Should include kinetic energy
        assert "speed=" in repr_str  # Should include speed


class TestParticleSystem:
    """Test cases for ParticleSystem class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = ParticleSystem()
        
        # Create test particles
        self.particle1 = Particle(
            mass=1000.0,
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3(10.0, 0.0, 0.0)
        )
        
        self.particle2 = Particle(
            mass=2000.0,
            position=Vector3(5.0, 0.0, 0.0),
            velocity=Vector3(-5.0, 0.0, 0.0)
        )
        
        self.particle3 = Particle(
            mass=500.0,
            position=Vector3(0.0, 5.0, 0.0),
            velocity=Vector3(0.0, -10.0, 0.0)
        )
    
    def test_system_initialization(self):
        """Test particle system initialization."""
        assert len(self.system) == 0
        assert len(self.system.particles) == 0
    
    def test_add_remove_particles(self):
        """Test adding and removing particles."""
        self.system.add_particle(self.particle1)
        assert len(self.system) == 1
        assert self.particle1 in self.system.particles
        
        self.system.add_particle(self.particle2)
        assert len(self.system) == 2
        
        # Test removal
        removed = self.system.remove_particle(self.particle1)
        assert removed is True
        assert len(self.system) == 1
        assert self.particle1 not in self.system.particles
        
        # Test removing non-existent particle
        removed = self.system.remove_particle(self.particle1)
        assert removed is False
    
    def test_get_active_particles(self):
        """Test getting active particles."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        self.system.add_particle(self.particle3)
        
        # All should be active initially
        active = self.system.get_active_particles()
        assert len(active) == 3
        
        # Deactivate one particle
        self.particle2.set_active(False)
        active = self.system.get_active_particles()
        assert len(active) == 2
        assert self.particle2 not in active
    
    def test_update_all_positions(self):
        """Test updating positions of all particles."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        
        initial_pos1 = Vector3(self.particle1.position.x, self.particle1.position.y, self.particle1.position.z)
        initial_pos2 = Vector3(self.particle2.position.x, self.particle2.position.y, self.particle2.position.z)
        
        dt = 0.1
        self.system.update_all_positions(dt)
        
        # Check that positions were updated
        expected_pos1 = initial_pos1 + (self.particle1.velocity * dt)
        expected_pos2 = initial_pos2 + (self.particle2.velocity * dt)
        
        assert self.particle1.position == expected_pos1
        assert self.particle2.position == expected_pos2
    
    def test_integrate_all_forces(self):
        """Test integrating forces for all particles."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        
        # Apply forces
        force1 = Vector3(1000.0, 0.0, 0.0)
        force2 = Vector3(-500.0, 0.0, 0.0)
        
        self.particle1.apply_force(force1, 0.1)
        self.particle2.apply_force(force2, 0.1)
        
        initial_vel1 = Vector3(self.particle1.velocity.x, self.particle1.velocity.y, self.particle1.velocity.z)
        initial_vel2 = Vector3(self.particle2.velocity.x, self.particle2.velocity.y, self.particle2.velocity.z)
        
        dt = 0.1
        self.system.integrate_all_forces(dt)
        
        # Check that velocities were updated
        expected_vel1 = initial_vel1 + (force1 / self.particle1.mass * dt)
        expected_vel2 = initial_vel2 + (force2 / self.particle2.mass * dt)
        
        assert self.particle1.velocity == expected_vel1
        assert self.particle2.velocity == expected_vel2
        
        # Forces should be reset
        assert self.particle1.get_accumulated_force() == Vector3.zero()
        assert self.particle2.get_accumulated_force() == Vector3.zero()
    
    def test_reset_all_forces(self):
        """Test resetting forces for all particles."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        
        # Apply forces
        self.particle1.apply_force(Vector3(100.0, 0.0, 0.0), 0.1)
        self.particle2.apply_force(Vector3(200.0, 0.0, 0.0), 0.1)
        
        # Verify forces are applied
        assert self.particle1.get_accumulated_force() != Vector3.zero()
        assert self.particle2.get_accumulated_force() != Vector3.zero()
        
        # Reset all forces
        self.system.reset_all_forces()
        
        # Verify forces are reset
        assert self.particle1.get_accumulated_force() == Vector3.zero()
        assert self.particle2.get_accumulated_force() == Vector3.zero()
    
    def test_total_kinetic_energy(self):
        """Test total kinetic energy calculation."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        self.system.add_particle(self.particle3)
        
        expected_total = (self.particle1.get_kinetic_energy() + 
                         self.particle2.get_kinetic_energy() + 
                         self.particle3.get_kinetic_energy())
        
        calculated_total = self.system.get_total_kinetic_energy()
        assert abs(calculated_total - expected_total) < 1e-6
        
        # Test with inactive particle
        self.particle2.set_active(False)
        expected_total_active = (self.particle1.get_kinetic_energy() + 
                               self.particle3.get_kinetic_energy())
        
        calculated_total_active = self.system.get_total_kinetic_energy()
        assert abs(calculated_total_active - expected_total_active) < 1e-6
    
    def test_center_of_mass(self):
        """Test center of mass calculation."""
        self.system.add_particle(self.particle1)  # mass=1000, pos=(0,0,0)
        self.system.add_particle(self.particle2)  # mass=2000, pos=(5,0,0)
        
        # COM = (m1*p1 + m2*p2) / (m1 + m2)
        total_mass = self.particle1.mass + self.particle2.mass
        weighted_pos = (self.particle1.position * self.particle1.mass + 
                       self.particle2.position * self.particle2.mass)
        expected_com = weighted_pos / total_mass
        
        calculated_com = self.system.get_center_of_mass()
        assert calculated_com == expected_com
    
    def test_center_of_mass_empty_system(self):
        """Test center of mass with empty system."""
        com = self.system.get_center_of_mass()
        assert com == Vector3.zero()
    
    def test_center_of_mass_zero_mass(self):
        """Test center of mass with zero mass particles."""
        zero_mass_particle = Particle(
            mass=0.0,
            position=Vector3(10.0, 10.0, 10.0),
            velocity=Vector3.zero()
        )
        
        self.system.add_particle(zero_mass_particle)
        com = self.system.get_center_of_mass()
        assert com == Vector3.zero()
    
    def test_system_clear(self):
        """Test clearing the particle system."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        
        assert len(self.system) == 2
        
        self.system.clear()
        assert len(self.system) == 0
        assert len(self.system.particles) == 0
    
    def test_system_iteration(self):
        """Test iterating over particles in system."""
        particles = [self.particle1, self.particle2, self.particle3]
        
        for particle in particles:
            self.system.add_particle(particle)
        
        # Test iteration
        iterated_particles = list(self.system)
        assert len(iterated_particles) == 3
        
        for particle in particles:
            assert particle in iterated_particles
    
    def test_string_representation(self):
        """Test string representation of ParticleSystem."""
        self.system.add_particle(self.particle1)
        self.system.add_particle(self.particle2)
        self.particle2.set_active(False)
        
        str_repr = str(self.system)
        assert "ParticleSystem" in str_repr
        assert "2 particles" in str_repr
        assert "1 active" in str_repr