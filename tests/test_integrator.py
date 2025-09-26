"""
Unit tests for PhysicsIntegrator class.

Tests integration methods, geodesic calculations, relativistic effects,
and compares different integration algorithms for accuracy and stability.
"""

import unittest
import math
from physics.integrator import PhysicsIntegrator, IntegrationState
from physics.vector3 import Vector3
from physics.black_hole import BlackHole
from physics.particle import Particle


class TestIntegrationState(unittest.TestCase):
    """Test the IntegrationState helper class."""
    
    def test_integration_state_creation(self):
        """Test creating integration states."""
        pos = Vector3(1.0, 2.0, 3.0)
        vel = Vector3(4.0, 5.0, 6.0)
        state = IntegrationState(pos, vel)
        
        self.assertEqual(state.position, pos)
        self.assertEqual(state.velocity, vel)
    
    def test_integration_state_addition(self):
        """Test adding integration states."""
        state1 = IntegrationState(Vector3(1.0, 2.0, 3.0), Vector3(4.0, 5.0, 6.0))
        state2 = IntegrationState(Vector3(2.0, 3.0, 4.0), Vector3(1.0, 1.0, 1.0))
        
        result = state1 + state2
        
        self.assertEqual(result.position, Vector3(3.0, 5.0, 7.0))
        self.assertEqual(result.velocity, Vector3(5.0, 6.0, 7.0))
    
    def test_integration_state_scalar_multiplication(self):
        """Test multiplying integration state by scalar."""
        state = IntegrationState(Vector3(2.0, 4.0, 6.0), Vector3(1.0, 3.0, 5.0))
        
        result = state * 2.0
        reverse_result = 2.0 * state
        
        expected_pos = Vector3(4.0, 8.0, 12.0)
        expected_vel = Vector3(2.0, 6.0, 10.0)
        
        self.assertEqual(result.position, expected_pos)
        self.assertEqual(result.velocity, expected_vel)
        self.assertEqual(reverse_result.position, expected_pos)
        self.assertEqual(reverse_result.velocity, expected_vel)


class TestPhysicsIntegrator(unittest.TestCase):
    """Test the PhysicsIntegrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integrator = PhysicsIntegrator()
        
        # Create a test black hole (solar mass)
        solar_mass = 1.989e30  # kg
        self.black_hole = BlackHole(mass=solar_mass, position=Vector3.zero())
        
        # Create test particles
        self.test_particle = Particle(
            mass=1000.0,  # kg
            position=Vector3(1e9, 0.0, 0.0),  # 1 million km from black hole
            velocity=Vector3(0.0, 1000.0, 0.0)  # 1 km/s tangential velocity
        )
        
        # Particle for circular orbit test
        orbital_distance = 1e10  # 10 million km
        orbital_velocity = math.sqrt(self.integrator.G * solar_mass / orbital_distance)
        self.circular_orbit_particle = Particle(
            mass=1000.0,
            position=Vector3(orbital_distance, 0.0, 0.0),
            velocity=Vector3(0.0, orbital_velocity, 0.0)
        )
    
    def test_integrator_initialization(self):
        """Test integrator initialization."""
        integrator = PhysicsIntegrator()
        
        self.assertEqual(integrator.G, 6.67430e-11)
        self.assertEqual(integrator.c, 299792458.0)
        self.assertTrue(integrator.enable_relativistic_effects)
        self.assertTrue(integrator.enable_geodesic_calculation)
        self.assertEqual(integrator.max_velocity_fraction, 0.1)
    
    def test_set_integration_parameters(self):
        """Test setting integration parameters."""
        self.integrator.set_integration_parameters(
            enable_relativistic=False,
            enable_geodesic=False,
            max_velocity_fraction=0.5
        )
        
        self.assertFalse(self.integrator.enable_relativistic_effects)
        self.assertFalse(self.integrator.enable_geodesic_calculation)
        self.assertEqual(self.integrator.max_velocity_fraction, 0.5)
    
    def test_velocity_limiting(self):
        """Test velocity limiting to fraction of speed of light."""
        # Create particle with very high velocity
        fast_particle = Particle(
            mass=1000.0,
            position=Vector3(1e9, 0.0, 0.0),
            velocity=Vector3(0.0, 0.5 * self.integrator.c, 0.0)  # 50% speed of light
        )
        
        # Apply velocity limiting
        limited_velocity = self.integrator._limit_velocity(fast_particle.velocity)
        
        max_allowed_speed = self.integrator.max_velocity_fraction * self.integrator.c
        self.assertLessEqual(limited_velocity.magnitude(), max_allowed_speed * 1.001)  # Small tolerance
    
    def test_calculate_derivative(self):
        """Test derivative calculation for integration."""
        state = IntegrationState(self.test_particle.position, self.test_particle.velocity)
        
        derivative = self.integrator._calculate_derivative(state, self.black_hole, self.test_particle.mass)
        
        # Velocity derivative should equal current velocity
        self.assertEqual(derivative.position, state.velocity)
        
        # Acceleration should point toward black hole
        acceleration_direction = derivative.velocity.normalize()
        expected_direction = (self.black_hole.position - state.position).normalize()
        
        # Check if acceleration points toward black hole (dot product should be close to 1)
        dot_product = acceleration_direction.dot(expected_direction)
        self.assertGreater(dot_product, 0.99)
    
    def test_geodesic_acceleration_vs_newtonian(self):
        """Test that geodesic acceleration reduces to Newtonian far from black hole."""
        # Disable relativistic effects for comparison
        self.integrator.set_integration_parameters(enable_relativistic=False)
        
        # Calculate geodesic acceleration
        geodesic_accel = self.integrator.calculate_geodesic_acceleration(
            self.test_particle.position, self.test_particle.velocity, self.black_hole
        )
        
        # Calculate Newtonian acceleration
        newtonian_force = self.black_hole.calculate_gravitational_force(
            self.test_particle.position, self.test_particle.mass
        )
        newtonian_accel = newtonian_force / self.test_particle.mass
        
        # Should be very close for distant particles
        difference = (geodesic_accel - newtonian_accel).magnitude()
        newtonian_magnitude = newtonian_accel.magnitude()
        
        self.assertLess(difference / newtonian_magnitude, 0.01)  # Within 1%
    
    def test_rk4_integration_single_step(self):
        """Test single step of RK4 integration."""
        particles = [self.test_particle.copy()]
        initial_position = Vector3(particles[0].position.x, particles[0].position.y, particles[0].position.z)
        initial_velocity = Vector3(particles[0].velocity.x, particles[0].velocity.y, particles[0].velocity.z)
        
        dt = 1.0  # 1 second
        self.integrator.integrate_rk4(particles, self.black_hole, dt)
        
        # Position should have changed
        self.assertNotEqual(particles[0].position, initial_position)
        
        # Velocity should have changed (due to gravitational acceleration)
        self.assertNotEqual(particles[0].velocity, initial_velocity)
        
        # Particle should still be active
        self.assertTrue(particles[0].active)
    
    def test_euler_integration_single_step(self):
        """Test single step of Euler integration."""
        particles = [self.test_particle.copy()]
        initial_position = Vector3(particles[0].position.x, particles[0].position.y, particles[0].position.z)
        
        dt = 1.0
        self.integrator.integrate_euler(particles, self.black_hole, dt)
        
        # Position should have changed
        self.assertNotEqual(particles[0].position, initial_position)
    
    def test_verlet_integration_single_step(self):
        """Test single step of Verlet integration."""
        particles = [self.test_particle.copy()]
        initial_position = Vector3(particles[0].position.x, particles[0].position.y, particles[0].position.z)
        
        dt = 1.0
        self.integrator.integrate_verlet(particles, self.black_hole, dt)
        
        # Position should have changed
        self.assertNotEqual(particles[0].position, initial_position)
    
    def test_integration_method_comparison(self):
        """Compare different integration methods for accuracy."""
        # Create identical particles for each method
        particle_rk4 = self.circular_orbit_particle.copy()
        particle_euler = self.circular_orbit_particle.copy()
        particle_verlet = self.circular_orbit_particle.copy()
        
        # Small time step for better accuracy comparison
        dt = 10.0  # 10 seconds
        num_steps = 100
        
        # Integrate using different methods
        for _ in range(num_steps):
            self.integrator.integrate_rk4([particle_rk4], self.black_hole, dt)
            self.integrator.integrate_euler([particle_euler], self.black_hole, dt)
            self.integrator.integrate_verlet([particle_verlet], self.black_hole, dt)
        
        # Calculate final distances from black hole
        distance_rk4 = particle_rk4.position.distance_to(self.black_hole.position)
        distance_euler = particle_euler.position.distance_to(self.black_hole.position)
        distance_verlet = particle_verlet.position.distance_to(self.black_hole.position)
        
        initial_distance = self.circular_orbit_particle.position.distance_to(self.black_hole.position)
        
        # RK4 should be most accurate (closest to initial distance for circular orbit)
        rk4_error = abs(distance_rk4 - initial_distance) / initial_distance
        euler_error = abs(distance_euler - initial_distance) / initial_distance
        verlet_error = abs(distance_verlet - initial_distance) / initial_distance
        
        # RK4 should have smallest error
        self.assertLess(rk4_error, euler_error)
        self.assertLess(rk4_error, 0.1)  # Should be within 10% for this test
    
    def test_energy_conservation(self):
        """Test energy conservation in orbital motion."""
        particle = self.circular_orbit_particle.copy()
        
        # Calculate initial energy
        initial_kinetic = particle.get_kinetic_energy()
        initial_potential = -self.integrator.G * self.black_hole.mass * particle.mass / \
                          particle.position.distance_to(self.black_hole.position)
        initial_total_energy = initial_kinetic + initial_potential
        
        # Integrate for several steps
        dt = 10.0
        for _ in range(50):
            self.integrator.integrate_rk4([particle], self.black_hole, dt)
        
        # Calculate final energy
        final_kinetic = particle.get_kinetic_energy()
        final_potential = -self.integrator.G * self.black_hole.mass * particle.mass / \
                         particle.position.distance_to(self.black_hole.position)
        final_total_energy = final_kinetic + final_potential
        
        # Energy should be approximately conserved
        energy_change = abs(final_total_energy - initial_total_energy)
        relative_change = energy_change / abs(initial_total_energy)
        
        self.assertLess(relative_change, 0.05)  # Within 5%
    
    def test_orbital_parameters_calculation(self):
        """Test calculation of orbital parameters."""
        params = self.integrator.calculate_orbital_parameters(self.circular_orbit_particle, self.black_hole)
        
        # Check that all expected parameters are present
        expected_keys = ['specific_energy', 'specific_angular_momentum', 'eccentricity',
                        'semi_major_axis', 'periapsis', 'apoapsis', 'is_bound',
                        'current_distance', 'current_speed']
        
        for key in expected_keys:
            self.assertIn(key, params)
        
        # For circular orbit, eccentricity should be close to 0
        self.assertLess(params['eccentricity'], 0.1)
        
        # Should be bound orbit (negative energy)
        self.assertTrue(params['is_bound'])
        self.assertLess(params['specific_energy'], 0)
    
    def test_relativistic_effects_application(self):
        """Test application of relativistic effects."""
        # Create particle close to black hole
        close_particle = Particle(
            mass=1000.0,
            position=Vector3(2.0 * self.black_hole.get_schwarzschild_radius(), 0.0, 0.0),
            velocity=Vector3(0.0, 0.05 * self.integrator.c, 0.0)  # 5% speed of light
        )
        
        initial_speed = close_particle.get_speed()
        
        # Apply relativistic effects
        self.integrator.apply_relativistic_effects(close_particle, self.black_hole)
        
        final_speed = close_particle.get_speed()
        
        # Speed should be affected by time dilation (reduced)
        self.assertLessEqual(final_speed, initial_speed)
    
    def test_inactive_particle_handling(self):
        """Test that inactive particles are not integrated."""
        particle = self.test_particle.copy()
        particle.set_active(False)
        
        initial_position = Vector3(particle.position.x, particle.position.y, particle.position.z)
        initial_velocity = Vector3(particle.velocity.x, particle.velocity.y, particle.velocity.z)
        
        # Try to integrate inactive particle
        self.integrator.integrate_rk4([particle], self.black_hole, 1.0)
        
        # Position and velocity should be unchanged
        self.assertEqual(particle.position, initial_position)
        self.assertEqual(particle.velocity, initial_velocity)
    
    def test_zero_mass_particle_handling(self):
        """Test handling of particles with zero mass."""
        zero_mass_particle = Particle(
            mass=0.0,
            position=Vector3(1e9, 0.0, 0.0),
            velocity=Vector3(0.0, 1000.0, 0.0)
        )
        
        # Should not crash with zero mass
        try:
            self.integrator.integrate_rk4([zero_mass_particle], self.black_hole, 1.0)
        except ZeroDivisionError:
            self.fail("Integration should handle zero mass particles gracefully")
    
    def test_extreme_proximity_handling(self):
        """Test handling of particles very close to black hole center."""
        # Particle very close to singularity
        close_particle = Particle(
            mass=1000.0,
            position=Vector3(1e-6, 0.0, 0.0),  # Very close to center
            velocity=Vector3(0.0, 1000.0, 0.0)
        )
        
        # Should not crash or produce NaN values
        try:
            self.integrator.integrate_rk4([close_particle], self.black_hole, 0.1)
            
            # Check for NaN values
            self.assertFalse(math.isnan(close_particle.position.x))
            self.assertFalse(math.isnan(close_particle.position.y))
            self.assertFalse(math.isnan(close_particle.position.z))
            self.assertFalse(math.isnan(close_particle.velocity.x))
            self.assertFalse(math.isnan(close_particle.velocity.y))
            self.assertFalse(math.isnan(close_particle.velocity.z))
            
        except Exception as e:
            self.fail(f"Integration should handle extreme proximity gracefully: {e}")
    
    def test_string_representation(self):
        """Test string representation of integrator."""
        str_repr = str(self.integrator)
        
        self.assertIn("PhysicsIntegrator", str_repr)
        self.assertIn("relativistic=True", str_repr)
        self.assertIn("geodesic=True", str_repr)
        self.assertIn("max_v=0.10c", str_repr)


class TestIntegrationAccuracy(unittest.TestCase):
    """Test integration accuracy with known analytical solutions."""
    
    def setUp(self):
        """Set up test fixtures for accuracy tests."""
        self.integrator = PhysicsIntegrator()
        # Disable relativistic effects for pure Newtonian comparison
        self.integrator.set_integration_parameters(enable_relativistic=False, enable_geodesic=False)
        
        # Create a smaller black hole for easier analytical comparison
        self.black_hole = BlackHole(mass=1e20, position=Vector3.zero())  # Much smaller than solar mass
    
    def test_free_fall_accuracy(self):
        """Test accuracy of free fall motion (analytical solution available)."""
        # Test that RK4 integration is more accurate than simple methods
        # by comparing energy conservation rather than exact analytical solutions
        
        # Particle starting with some orbital velocity
        initial_distance = 1e6  # 1000 km
        particle = Particle(
            mass=1000.0,
            position=Vector3(initial_distance, 0.0, 0.0),
            velocity=Vector3(0.0, 100.0, 0.0)  # Small tangential velocity
        )
        
        # Calculate initial total energy
        initial_kinetic = particle.get_kinetic_energy()
        initial_potential = -self.integrator.G * self.black_hole.mass * particle.mass / initial_distance
        initial_total_energy = initial_kinetic + initial_potential
        
        # Integrate for several steps with small time step
        dt = 0.5
        for _ in range(100):
            self.integrator.integrate_rk4([particle], self.black_hole, dt)
        
        # Calculate final total energy
        final_distance = particle.position.distance_to(self.black_hole.position)
        final_kinetic = particle.get_kinetic_energy()
        final_potential = -self.integrator.G * self.black_hole.mass * particle.mass / final_distance
        final_total_energy = final_kinetic + final_potential
        
        # Energy should be conserved (within reasonable tolerance for numerical integration)
        energy_change = abs(final_total_energy - initial_total_energy)
        relative_energy_change = energy_change / abs(initial_total_energy)
        
        # RK4 should conserve energy reasonably well
        self.assertLess(relative_energy_change, 0.1)  # Within 10%


if __name__ == '__main__':
    unittest.main()