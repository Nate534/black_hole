"""
Configuration data models for the black hole simulation.

This module provides dataclasses for render and physics configuration
with validation methods to ensure parameter correctness.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RenderConfig:
    """
    Configuration for rendering settings.
    
    Controls window size, frame rate, visual quality, and rendering options.
    """
    window_width: int = 1920
    window_height: int = 1080
    target_fps: int = 60
    vsync_enabled: bool = True
    particle_size: float = 2.0
    trail_length: int = 100
    
    def validate(self) -> Dict[str, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
        """
        errors = {}
        
        # Window dimensions validation
        if self.window_width <= 0:
            errors['window_width'] = "Window width must be positive"
        elif self.window_width < 640:
            errors['window_width'] = "Window width should be at least 640 pixels"
        elif self.window_width > 7680:  # 8K width
            errors['window_width'] = "Window width exceeds reasonable maximum (7680)"
            
        if self.window_height <= 0:
            errors['window_height'] = "Window height must be positive"
        elif self.window_height < 480:
            errors['window_height'] = "Window height should be at least 480 pixels"
        elif self.window_height > 4320:  # 8K height
            errors['window_height'] = "Window height exceeds reasonable maximum (4320)"
        
        # Frame rate validation
        if self.target_fps <= 0:
            errors['target_fps'] = "Target FPS must be positive"
        elif self.target_fps > 240:
            errors['target_fps'] = "Target FPS exceeds reasonable maximum (240)"
        
        # Particle size validation
        if self.particle_size <= 0.0:
            errors['particle_size'] = "Particle size must be positive"
        elif self.particle_size > 50.0:
            errors['particle_size'] = "Particle size exceeds reasonable maximum (50.0)"
        
        # Trail length validation
        if self.trail_length < 0:
            errors['trail_length'] = "Trail length cannot be negative"
        elif self.trail_length > 10000:
            errors['trail_length'] = "Trail length exceeds reasonable maximum (10000)"
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the configuration is valid.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return len(self.validate()) == 0
    
    def get_aspect_ratio(self) -> float:
        """
        Calculate the aspect ratio of the window.
        
        Returns:
            float: Width/height aspect ratio
        """
        return self.window_width / self.window_height if self.window_height > 0 else 1.0
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'RenderConfig':
        """
        Create a configuration from a preset.
        
        Args:
            preset_name: Name of the preset ('low', 'medium', 'high', 'ultra')
            
        Returns:
            RenderConfig: Configuration with preset values
            
        Raises:
            ValueError: If preset name is not recognized
        """
        presets = {
            'low': cls(
                window_width=1280,
                window_height=720,
                target_fps=30,
                vsync_enabled=True,
                particle_size=1.5,
                trail_length=50
            ),
            'medium': cls(
                window_width=1920,
                window_height=1080,
                target_fps=60,
                vsync_enabled=True,
                particle_size=2.0,
                trail_length=100
            ),
            'high': cls(
                window_width=2560,
                window_height=1440,
                target_fps=60,
                vsync_enabled=True,
                particle_size=2.5,
                trail_length=200
            ),
            'ultra': cls(
                window_width=3840,
                window_height=2160,
                target_fps=60,
                vsync_enabled=False,
                particle_size=3.0,
                trail_length=500
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return presets[preset_name]


@dataclass
class PhysicsConfig:
    """
    Configuration for physics simulation settings.
    
    Controls physical constants, simulation parameters, and integration settings.
    """
    gravitational_constant: float = 6.67430e-11  # m³/kg⋅s²
    speed_of_light: float = 299792458.0  # m/s
    time_step: float = 0.016  # ~60 FPS (1/60 seconds)
    max_particles: int = 10000
    integration_method: str = "rk4"
    
    def validate(self) -> Dict[str, str]:
        """
        Validate physics configuration parameters.
        
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
        """
        errors = {}
        
        # Gravitational constant validation
        if self.gravitational_constant <= 0.0:
            errors['gravitational_constant'] = "Gravitational constant must be positive"
        elif self.gravitational_constant > 1e-5:  # Unreasonably large
            errors['gravitational_constant'] = "Gravitational constant seems unreasonably large"
        
        # Speed of light validation
        if self.speed_of_light <= 0.0:
            errors['speed_of_light'] = "Speed of light must be positive"
        elif self.speed_of_light > 1e12:  # Unreasonably large
            errors['speed_of_light'] = "Speed of light seems unreasonably large"
        
        # Time step validation
        if self.time_step <= 0.0:
            errors['time_step'] = "Time step must be positive"
        elif self.time_step > 1.0:
            errors['time_step'] = "Time step is too large (should be < 1.0 second)"
        elif self.time_step < 1e-6:
            errors['time_step'] = "Time step is too small (should be >= 1e-6 second)"
        
        # Max particles validation
        if self.max_particles <= 0:
            errors['max_particles'] = "Maximum particles must be positive"
        elif self.max_particles > 1000000:
            errors['max_particles'] = "Maximum particles exceeds reasonable limit (1,000,000)"
        
        # Integration method validation
        valid_methods = ["euler", "rk2", "rk4", "verlet"]
        if self.integration_method not in valid_methods:
            errors['integration_method'] = f"Integration method must be one of: {valid_methods}"
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the physics configuration is valid.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return len(self.validate()) == 0
    
    def get_target_fps(self) -> float:
        """
        Calculate the target FPS based on time step.
        
        Returns:
            float: Target frames per second
        """
        return 1.0 / self.time_step if self.time_step > 0 else 60.0
    
    def set_target_fps(self, fps: float) -> None:
        """
        Set the time step based on target FPS.
        
        Args:
            fps: Target frames per second
            
        Raises:
            ValueError: If FPS is not positive
        """
        if fps <= 0:
            raise ValueError("FPS must be positive")
        self.time_step = 1.0 / fps
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'PhysicsConfig':
        """
        Create a physics configuration from a preset.
        
        Args:
            preset_name: Name of the preset ('fast', 'balanced', 'accurate')
            
        Returns:
            PhysicsConfig: Configuration with preset values
            
        Raises:
            ValueError: If preset name is not recognized
        """
        presets = {
            'fast': cls(
                time_step=0.033,  # ~30 FPS
                max_particles=5000,
                integration_method="euler"
            ),
            'balanced': cls(
                time_step=0.016,  # ~60 FPS
                max_particles=10000,
                integration_method="rk4"
            ),
            'accurate': cls(
                time_step=0.008,  # ~120 FPS
                max_particles=20000,
                integration_method="rk4"
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return presets[preset_name]


@dataclass
class SimulationConfig:
    """
    Combined configuration for the entire simulation.
    
    Contains both render and physics configurations with convenience methods.
    """
    render: RenderConfig
    physics: PhysicsConfig
    
    def __init__(self, render: Optional[RenderConfig] = None, physics: Optional[PhysicsConfig] = None):
        """
        Initialize simulation configuration.
        
        Args:
            render: Render configuration (uses default if None)
            physics: Physics configuration (uses default if None)
        """
        self.render = render if render is not None else RenderConfig()
        self.physics = physics if physics is not None else PhysicsConfig()
    
    def validate(self) -> Dict[str, Dict[str, str]]:
        """
        Validate both render and physics configurations.
        
        Returns:
            Dict[str, Dict[str, str]]: Nested dictionary of validation errors
        """
        return {
            'render': self.render.validate(),
            'physics': self.physics.validate()
        }
    
    def is_valid(self) -> bool:
        """
        Check if the entire simulation configuration is valid.
        
        Returns:
            bool: True if both configurations are valid, False otherwise
        """
        validation_results = self.validate()
        return (len(validation_results['render']) == 0 and 
                len(validation_results['physics']) == 0)
    
    def get_validation_summary(self) -> str:
        """
        Get a human-readable summary of validation errors.
        
        Returns:
            str: Summary of validation errors, or "Valid" if no errors
        """
        validation_results = self.validate()
        errors = []
        
        for category, category_errors in validation_results.items():
            for field, error in category_errors.items():
                errors.append(f"{category}.{field}: {error}")
        
        return "Valid" if not errors else "\n".join(errors)
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'SimulationConfig':
        """
        Create a simulation configuration from a preset.
        
        Args:
            preset_name: Name of the preset ('low', 'medium', 'high', 'ultra')
            
        Returns:
            SimulationConfig: Configuration with preset values
        """
        render_config = RenderConfig.create_preset(preset_name)
        
        # Map render presets to physics presets
        physics_preset_map = {
            'low': 'fast',
            'medium': 'balanced', 
            'high': 'balanced',
            'ultra': 'accurate'
        }
        
        physics_preset = physics_preset_map.get(preset_name, 'balanced')
        physics_config = PhysicsConfig.create_preset(physics_preset)
        
        return cls(render=render_config, physics=physics_config)