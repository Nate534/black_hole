"""
Configuration settings for the black hole simulation.

This module provides comprehensive configuration management with validation,
presets, and error handling for all simulation parameters.
"""

# Import the comprehensive configuration models
from config_models import RenderConfig, PhysicsConfig, SimulationConfig

# Re-export for backward compatibility and convenience
__all__ = ['RenderConfig', 'PhysicsConfig', 'SimulationConfig']


def get_default_config() -> SimulationConfig:
    """
    Get the default simulation configuration.
    
    Returns:
        SimulationConfig: Default configuration with balanced settings
    """
    return SimulationConfig()


def get_preset_config(preset_name: str) -> SimulationConfig:
    """
    Get a preset simulation configuration.
    
    Args:
        preset_name: Name of the preset ('low', 'medium', 'high', 'ultra')
        
    Returns:
        SimulationConfig: Configuration with preset values
        
    Raises:
        ValueError: If preset name is not recognized
    """
    return SimulationConfig.create_preset(preset_name)


def validate_config(config: SimulationConfig) -> str:
    """
    Validate a simulation configuration and return a summary.
    
    Args:
        config: The configuration to validate
        
    Returns:
        str: Validation summary ("Valid" if no errors, otherwise error details)
    """
    return config.get_validation_summary()