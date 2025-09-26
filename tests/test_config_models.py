"""
Unit tests for configuration data models.

Tests validation methods, presets, and error conditions for
RenderConfig, PhysicsConfig, and SimulationConfig classes.
"""

import unittest
from config_models import RenderConfig, PhysicsConfig, SimulationConfig


class TestRenderConfig(unittest.TestCase):
    """Test cases for RenderConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.default_config = RenderConfig()
        self.valid_config = RenderConfig(
            window_width=1920,
            window_height=1080,
            target_fps=60,
            vsync_enabled=True,
            particle_size=2.0,
            trail_length=100
        )
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = RenderConfig()
        self.assertEqual(config.window_width, 1920)
        self.assertEqual(config.window_height, 1080)
        self.assertEqual(config.target_fps, 60)
        self.assertTrue(config.vsync_enabled)
        self.assertEqual(config.particle_size, 2.0)
        self.assertEqual(config.trail_length, 100)
    
    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        errors = self.valid_config.validate()
        self.assertEqual(len(errors), 0)
        self.assertTrue(self.valid_config.is_valid())
    
    def test_invalid_window_dimensions(self):
        """Test validation of invalid window dimensions."""
        # Negative width
        config = RenderConfig(window_width=-100, window_height=1080)
        errors = config.validate()
        self.assertIn('window_width', errors)
        self.assertFalse(config.is_valid())
        
        # Too small width
        config = RenderConfig(window_width=500, window_height=1080)
        errors = config.validate()
        self.assertIn('window_width', errors)
        
        # Too large width
        config = RenderConfig(window_width=10000, window_height=1080)
        errors = config.validate()
        self.assertIn('window_width', errors)
        
        # Negative height
        config = RenderConfig(window_width=1920, window_height=-100)
        errors = config.validate()
        self.assertIn('window_height', errors)
        
        # Too small height
        config = RenderConfig(window_width=1920, window_height=400)
        errors = config.validate()
        self.assertIn('window_height', errors)
        
        # Too large height
        config = RenderConfig(window_width=1920, window_height=5000)
        errors = config.validate()
        self.assertIn('window_height', errors)
    
    def test_invalid_fps(self):
        """Test validation of invalid FPS values."""
        # Negative FPS
        config = RenderConfig(target_fps=-10)
        errors = config.validate()
        self.assertIn('target_fps', errors)
        
        # Zero FPS
        config = RenderConfig(target_fps=0)
        errors = config.validate()
        self.assertIn('target_fps', errors)
        
        # Too high FPS
        config = RenderConfig(target_fps=500)
        errors = config.validate()
        self.assertIn('target_fps', errors)
    
    def test_invalid_particle_size(self):
        """Test validation of invalid particle size."""
        # Negative size
        config = RenderConfig(particle_size=-1.0)
        errors = config.validate()
        self.assertIn('particle_size', errors)
        
        # Zero size
        config = RenderConfig(particle_size=0.0)
        errors = config.validate()
        self.assertIn('particle_size', errors)
        
        # Too large size
        config = RenderConfig(particle_size=100.0)
        errors = config.validate()
        self.assertIn('particle_size', errors)
    
    def test_invalid_trail_length(self):
        """Test validation of invalid trail length."""
        # Negative trail length
        config = RenderConfig(trail_length=-10)
        errors = config.validate()
        self.assertIn('trail_length', errors)
        
        # Too large trail length
        config = RenderConfig(trail_length=50000)
        errors = config.validate()
        self.assertIn('trail_length', errors)
    
    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        config = RenderConfig(window_width=1920, window_height=1080)
        self.assertAlmostEqual(config.get_aspect_ratio(), 16/9, places=5)
        
        config = RenderConfig(window_width=1280, window_height=720)
        self.assertAlmostEqual(config.get_aspect_ratio(), 16/9, places=5)
        
        # Edge case: zero height
        config = RenderConfig(window_width=1920, window_height=0)
        self.assertEqual(config.get_aspect_ratio(), 1.0)
    
    def test_presets(self):
        """Test configuration presets."""
        # Test all valid presets
        presets = ['low', 'medium', 'high', 'ultra']
        for preset in presets:
            config = RenderConfig.create_preset(preset)
            self.assertTrue(config.is_valid())
        
        # Test specific preset values
        low_config = RenderConfig.create_preset('low')
        self.assertEqual(low_config.window_width, 1280)
        self.assertEqual(low_config.window_height, 720)
        self.assertEqual(low_config.target_fps, 30)
        
        ultra_config = RenderConfig.create_preset('ultra')
        self.assertEqual(ultra_config.window_width, 3840)
        self.assertEqual(ultra_config.window_height, 2160)
        self.assertFalse(ultra_config.vsync_enabled)
    
    def test_invalid_preset(self):
        """Test invalid preset name."""
        with self.assertRaises(ValueError):
            RenderConfig.create_preset('invalid')


class TestPhysicsConfig(unittest.TestCase):
    """Test cases for PhysicsConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.default_config = PhysicsConfig()
        self.valid_config = PhysicsConfig(
            gravitational_constant=6.67430e-11,
            speed_of_light=299792458.0,
            time_step=0.016,
            max_particles=10000,
            integration_method="rk4"
        )
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = PhysicsConfig()
        self.assertEqual(config.gravitational_constant, 6.67430e-11)
        self.assertEqual(config.speed_of_light, 299792458.0)
        self.assertEqual(config.time_step, 0.016)
        self.assertEqual(config.max_particles, 10000)
        self.assertEqual(config.integration_method, "rk4")
    
    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        errors = self.valid_config.validate()
        self.assertEqual(len(errors), 0)
        self.assertTrue(self.valid_config.is_valid())
    
    def test_invalid_gravitational_constant(self):
        """Test validation of invalid gravitational constant."""
        # Negative value
        config = PhysicsConfig(gravitational_constant=-1e-11)
        errors = config.validate()
        self.assertIn('gravitational_constant', errors)
        
        # Zero value
        config = PhysicsConfig(gravitational_constant=0.0)
        errors = config.validate()
        self.assertIn('gravitational_constant', errors)
        
        # Too large value
        config = PhysicsConfig(gravitational_constant=1.0)
        errors = config.validate()
        self.assertIn('gravitational_constant', errors)
    
    def test_invalid_speed_of_light(self):
        """Test validation of invalid speed of light."""
        # Negative value
        config = PhysicsConfig(speed_of_light=-1e8)
        errors = config.validate()
        self.assertIn('speed_of_light', errors)
        
        # Zero value
        config = PhysicsConfig(speed_of_light=0.0)
        errors = config.validate()
        self.assertIn('speed_of_light', errors)
        
        # Too large value
        config = PhysicsConfig(speed_of_light=1e15)
        errors = config.validate()
        self.assertIn('speed_of_light', errors)
    
    def test_invalid_time_step(self):
        """Test validation of invalid time step."""
        # Negative value
        config = PhysicsConfig(time_step=-0.01)
        errors = config.validate()
        self.assertIn('time_step', errors)
        
        # Zero value
        config = PhysicsConfig(time_step=0.0)
        errors = config.validate()
        self.assertIn('time_step', errors)
        
        # Too large value
        config = PhysicsConfig(time_step=2.0)
        errors = config.validate()
        self.assertIn('time_step', errors)
        
        # Too small value
        config = PhysicsConfig(time_step=1e-7)
        errors = config.validate()
        self.assertIn('time_step', errors)
    
    def test_invalid_max_particles(self):
        """Test validation of invalid max particles."""
        # Negative value
        config = PhysicsConfig(max_particles=-100)
        errors = config.validate()
        self.assertIn('max_particles', errors)
        
        # Zero value
        config = PhysicsConfig(max_particles=0)
        errors = config.validate()
        self.assertIn('max_particles', errors)
        
        # Too large value
        config = PhysicsConfig(max_particles=2000000)
        errors = config.validate()
        self.assertIn('max_particles', errors)
    
    def test_invalid_integration_method(self):
        """Test validation of invalid integration method."""
        config = PhysicsConfig(integration_method="invalid_method")
        errors = config.validate()
        self.assertIn('integration_method', errors)
        
        # Test valid methods don't produce errors
        valid_methods = ["euler", "rk2", "rk4", "verlet"]
        for method in valid_methods:
            config = PhysicsConfig(integration_method=method)
            errors = config.validate()
            self.assertNotIn('integration_method', errors)
    
    def test_fps_methods(self):
        """Test FPS-related methods."""
        config = PhysicsConfig(time_step=0.016)
        fps = config.get_target_fps()
        self.assertAlmostEqual(fps, 62.5, places=1)
        
        # Test setting FPS
        config.set_target_fps(30.0)
        self.assertAlmostEqual(config.time_step, 1.0/30.0, places=6)
        
        # Test invalid FPS
        with self.assertRaises(ValueError):
            config.set_target_fps(-10.0)
        
        with self.assertRaises(ValueError):
            config.set_target_fps(0.0)
    
    def test_presets(self):
        """Test physics configuration presets."""
        # Test all valid presets
        presets = ['fast', 'balanced', 'accurate']
        for preset in presets:
            config = PhysicsConfig.create_preset(preset)
            self.assertTrue(config.is_valid())
        
        # Test specific preset values
        fast_config = PhysicsConfig.create_preset('fast')
        self.assertEqual(fast_config.time_step, 0.033)
        self.assertEqual(fast_config.max_particles, 5000)
        self.assertEqual(fast_config.integration_method, "euler")
        
        accurate_config = PhysicsConfig.create_preset('accurate')
        self.assertEqual(accurate_config.time_step, 0.008)
        self.assertEqual(accurate_config.max_particles, 20000)
        self.assertEqual(accurate_config.integration_method, "rk4")
    
    def test_invalid_preset(self):
        """Test invalid preset name."""
        with self.assertRaises(ValueError):
            PhysicsConfig.create_preset('invalid')


class TestSimulationConfig(unittest.TestCase):
    """Test cases for SimulationConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.render_config = RenderConfig()
        self.physics_config = PhysicsConfig()
        self.simulation_config = SimulationConfig(self.render_config, self.physics_config)
    
    def test_initialization_with_configs(self):
        """Test initialization with provided configurations."""
        config = SimulationConfig(self.render_config, self.physics_config)
        self.assertEqual(config.render, self.render_config)
        self.assertEqual(config.physics, self.physics_config)
    
    def test_initialization_with_defaults(self):
        """Test initialization with default configurations."""
        config = SimulationConfig()
        self.assertIsInstance(config.render, RenderConfig)
        self.assertIsInstance(config.physics, PhysicsConfig)
        self.assertTrue(config.render.is_valid())
        self.assertTrue(config.physics.is_valid())
    
    def test_initialization_with_partial_configs(self):
        """Test initialization with only one configuration provided."""
        # Only render config
        config = SimulationConfig(render=self.render_config)
        self.assertEqual(config.render, self.render_config)
        self.assertIsInstance(config.physics, PhysicsConfig)
        
        # Only physics config
        config = SimulationConfig(physics=self.physics_config)
        self.assertIsInstance(config.render, RenderConfig)
        self.assertEqual(config.physics, self.physics_config)
    
    def test_validation(self):
        """Test validation of simulation configuration."""
        # Valid configuration
        validation_results = self.simulation_config.validate()
        self.assertEqual(len(validation_results['render']), 0)
        self.assertEqual(len(validation_results['physics']), 0)
        self.assertTrue(self.simulation_config.is_valid())
        
        # Invalid render configuration
        invalid_render = RenderConfig(window_width=-100)
        config = SimulationConfig(render=invalid_render, physics=self.physics_config)
        validation_results = config.validate()
        self.assertGreater(len(validation_results['render']), 0)
        self.assertEqual(len(validation_results['physics']), 0)
        self.assertFalse(config.is_valid())
        
        # Invalid physics configuration
        invalid_physics = PhysicsConfig(time_step=-0.01)
        config = SimulationConfig(render=self.render_config, physics=invalid_physics)
        validation_results = config.validate()
        self.assertEqual(len(validation_results['render']), 0)
        self.assertGreater(len(validation_results['physics']), 0)
        self.assertFalse(config.is_valid())
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        # Valid configuration
        summary = self.simulation_config.get_validation_summary()
        self.assertEqual(summary, "Valid")
        
        # Invalid configuration
        invalid_render = RenderConfig(window_width=-100, target_fps=0)
        invalid_physics = PhysicsConfig(time_step=-0.01)
        config = SimulationConfig(render=invalid_render, physics=invalid_physics)
        summary = config.get_validation_summary()
        
        self.assertIn("render.window_width", summary)
        self.assertIn("render.target_fps", summary)
        self.assertIn("physics.time_step", summary)
        self.assertNotEqual(summary, "Valid")
    
    def test_presets(self):
        """Test simulation configuration presets."""
        # Test all valid presets
        presets = ['low', 'medium', 'high', 'ultra']
        for preset in presets:
            config = SimulationConfig.create_preset(preset)
            self.assertTrue(config.is_valid())
            self.assertIsInstance(config.render, RenderConfig)
            self.assertIsInstance(config.physics, PhysicsConfig)
        
        # Test specific preset mapping
        low_config = SimulationConfig.create_preset('low')
        self.assertEqual(low_config.render.window_width, 1280)
        self.assertEqual(low_config.physics.integration_method, "euler")
        
        ultra_config = SimulationConfig.create_preset('ultra')
        self.assertEqual(ultra_config.render.window_width, 3840)
        self.assertEqual(ultra_config.physics.integration_method, "rk4")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values explicitly
        config = SimulationConfig(render=None, physics=None)
        self.assertIsInstance(config.render, RenderConfig)
        self.assertIsInstance(config.physics, PhysicsConfig)
        self.assertTrue(config.is_valid())


if __name__ == '__main__':
    unittest.main()