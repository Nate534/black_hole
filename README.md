# Black Hole Simulation

A real-time 3D black hole simulation built with Python and Pygame, featuring gravitational physics, particle systems, and visual effects.

## Features

- **Realistic Physics**: Implements gravitational acceleration and Schwarzschild radius calculations
- **Particle System**: Simulates accretion disk with hundreds of particles
- **Visual Effects**: Glow effects, particle trails, and gravitational lensing visualization
- **Interactive Camera**: Full 3D camera controls with zoom and movement
- **Real-time Simulation**: 60 FPS simulation with dynamic particle generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/black-hole-simulation.git
cd black-hole-simulation
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python src/main.py
```

### Controls

- **WASD**: Move camera
- **Q/E**: Move camera up/down
- **+/-**: Zoom in/out
- **R**: Reset camera position
- **G**: Toggle gravity on/off
- **ESC**: Exit simulation

## Project Structure

```
src/
├── main.py              # Main simulation loop
├── physics/             # Physics calculations
│   ├── black_hole.py    # Black hole physics
│   ├── particle.py      # Particle system
│   └── constants.py     # Physical constants
├── rendering/           # Rendering system
│   ├── renderer.py      # Main renderer
│   ├── camera.py        # Camera controls
│   └── camera_config.py # Camera settings
└── utils/               # Utility functions
    └── vectors.py       # Vector operations
```

## Physics

The simulation implements:
- **Gravitational Force**: F = GMm/r²
- **Schwarzschild Radius**: Rs = 2GM/c²
- **Event Horizon**: Particles crossing the event horizon turn black
- **Orbital Mechanics**: Particles follow realistic orbital paths

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Physics calculations based on Einstein's General Relativity
- Inspired by real black hole observations from Event Horizon Telescope
- Built with Python, Pygame, and NumPy