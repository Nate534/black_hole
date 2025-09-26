# Black Hole Simulation

A real-time black hole simulation demonstrating gravitational effects, particle dynamics, and gravitational lensing using Python and OpenGL.

## Project Structure

```
black-hole-simulation/
├── display/              # Window management, input handling, UI controls
├── physics/              # Simulation logic, mathematical models, integration
├── gpu_rendering/        # OpenGL resources, shaders, GPU compute operations
├── venv/                 # Python virtual environment
├── main.py              # Main application entry point
├── config.py            # Configuration settings and validation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

1. **Create and activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation:**
   ```bash
   python main.py
   ```

## Requirements

- Python 3.8+
- OpenGL 4.3+ (for compute shaders)
- Graphics card with OpenGL support

## Dependencies

- **NumPy**: Mathematical operations and arrays
- **PyOpenGL**: OpenGL bindings for Python
- **GLFW**: Window and input management
- **ImGui**: Immediate mode GUI for controls
- **Pytest**: Testing framework
- **Numba**: JIT compilation for performance (optional)

## Development

The project follows a modular architecture with clear separation of concerns:

- **Display Module**: Handles windowing, input processing, and UI rendering
- **Physics Module**: Contains black hole physics, particle dynamics, and integration algorithms
- **GPU Rendering Module**: Manages OpenGL resources and GPU compute operations

## License

This project is for educational purposes demonstrating physics simulation and GPU programming concepts.