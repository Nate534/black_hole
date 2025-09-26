#!/usr/bin/env python3
"""
Black Hole Simulation - Main Application Entry Point

A real-time black hole simulation demonstrating gravitational effects,
particle dynamics, and gravitational lensing using Python and OpenGL.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simulation.log')
        ]
    )


def main():
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Black Hole Simulation")
    
    try:
        # TODO: Initialize modules and start simulation loop
        # This will be implemented in subsequent tasks
        logger.info("Simulation setup complete - ready for implementation")
        print("Black Hole Simulation - Project structure initialized")
        print("Run with virtual environment activated:")
        print("  Windows: venv\\Scripts\\activate")
        print("  Linux/Mac: source venv/bin/activate")
        print("  Then: pip install -r requirements.txt")
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()