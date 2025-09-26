# Design Document

## Overview

The black hole simulation will be implemented as a modular Python application using OpenGL for rendering and GPU compute shaders for physics calculations. The architecture separates concerns into three main modules: display management, physics simulation, and GPU rendering. The system will use modern OpenGL features including compute shaders for parallel particle trajectory calculations and vertex buffer objects for efficient rendering.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Display     │    │     Physics     │    │  GPU Rendering  │
│                 │    │                 │    │                 │
│ - Window Mgmt   │◄──►│ - BlackHole     │◄──►│ - Shaders       │
│ - Input Handler │    │ - Particle      │    │ - Buffers       │
│ - UI Controls   │    │ - Integrator    │    │ - Compute       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Main Loop     │
                    │                 │
                    │ - Coordination  │
                    │ - Timing        │
                    │ - State Mgmt    │
                    └─────────────────┘
```

### Module Dependencies

- **Display Module**: Handles windowing (GLFW), input processing, and UI rendering
- **Physics Module**: Contains simulation logic, mathematical models, and integration algorithms  
- **GPU Rendering Module**: Manages OpenGL resources, shaders, and GPU compute operations
- **Main Application**: Coordinates between modules and manages the simulation loop

## Components and Interfaces

### Display Module (`display/`)

**WindowManager Class**
```python
class WindowManager:
    def create_window(width: int, height: int) -> bool
    def should_close() -> bool
    def swap_buffers() -> None
    def poll_events() -> None
    def get_framebuffer_size() -> Tuple[int, int]
```

**InputHandler Class**
```python
class InputHandler:
    def process_keyboard(window, key: int, action: int) -> None
    def process_mouse(window, button: int, action: int) -> None
    def get_mouse_position() -> Tuple[float, float]
    def is_key_pressed(key: int) -> bool
```

**UIController Class**
```python
class UIController:
    def render_controls() -> None
    def get_black_hole_mass() -> float
    def get_particle_spawn_params() -> ParticleParams
    def handle_parameter_changes() -> Dict[str, Any]
```

### Physics Module (`physics/`)

**BlackHole Class**
```python
class BlackHole:
    def __init__(mass: float, position: Vector3)
    def get_schwarzschild_radius() -> float
    def get_photon_sphere_radius() -> float
    def calculate_gravitational_force(particle_pos: Vector3) -> Vector3
    def is_within_event_horizon(particle_pos: Vector3) -> bool
    def get_spacetime_curvature(position: Vector3) -> float
```

**Particle Class**
```python
class Particle:
    def __init__(mass: float, position: Vector3, velocity: Vector3)
    def update_position(dt: float) -> None
    def apply_force(force: Vector3, dt: float) -> None
    def get_kinetic_energy() -> float
    def is_active() -> bool
```

**PhysicsIntegrator Class**
```python
class PhysicsIntegrator:
    def integrate_rk4(particles: List[Particle], black_hole: BlackHole, dt: float) -> None
    def calculate_geodesic(particle: Particle, black_hole: BlackHole) -> Vector3
    def apply_relativistic_effects(particle: Particle, black_hole: BlackHole) -> None
```

### GPU Rendering Module (`gpu_rendering/`)

**ShaderManager Class**
```python
class ShaderManager:
    def load_compute_shader(filepath: str) -> int
    def load_vertex_fragment_shaders(vertex_path: str, fragment_path: str) -> int
    def use_program(program_id: int) -> None
    def set_uniform(name: str, value: Any) -> None
```

**BufferManager Class**
```python
class BufferManager:
    def create_particle_buffer(particles: List[Particle]) -> int
    def update_particle_buffer(buffer_id: int, particles: List[Particle]) -> None
    def create_vertex_array() -> int
    def bind_buffer(buffer_id: int) -> None
```

**ComputeRenderer Class**
```python
class ComputeRenderer:
    def dispatch_particle_update(num_particles: int) -> None
    def synchronize_gpu() -> None
    def setup_compute_uniforms(black_hole: BlackHole, dt: float) -> None
```

## Data Models

### Core Data Structures

**Vector3**
```python
@dataclass
class Vector3:
    x: float
    y: float  
    z: float
    
    def magnitude() -> float
    def normalize() -> 'Vector3'
    def dot(other: 'Vector3') -> float
    def cross(other: 'Vector3') -> 'Vector3'
```

**ParticleData (GPU Buffer Layout)**
```python
@dataclass
class ParticleData:
    position: Vector3      # 12 bytes
    velocity: Vector3      # 12 bytes  
    mass: float           # 4 bytes
    active: int           # 4 bytes (boolean as int)
    # Total: 32 bytes (aligned)
```

**SimulationState**
```python
@dataclass
class SimulationState:
    black_hole: BlackHole
    particles: List[Particle]
    camera_position: Vector3
    camera_rotation: Vector3
    time_step: float
    simulation_speed: float
    paused: bool
```

### Configuration Models

**RenderConfig**
```python
@dataclass
class RenderConfig:
    window_width: int = 1920
    window_height: int = 1080
    target_fps: int = 60
    vsync_enabled: bool = True
    particle_size: float = 2.0
    trail_length: int = 100
```

**PhysicsConfig**
```python
@dataclass  
class PhysicsConfig:
    gravitational_constant: float = 6.67430e-11
    speed_of_light: float = 299792458.0
    time_step: float = 0.016  # ~60 FPS
    max_particles: int = 10000
    integration_method: str = "rk4"
```

## Error Handling

### OpenGL Error Management
- Implement OpenGL error checking after each GL call in debug mode
- Use context managers for resource cleanup (shaders, buffers, textures)
- Graceful degradation when compute shaders are not supported

### Physics Error Handling
- Validate particle positions for NaN/infinity values
- Handle singularity approaches with appropriate limits
- Implement bounds checking for simulation space

### Resource Management
- Automatic cleanup of GPU resources using context managers
- Memory pool for particle allocation to avoid frequent allocations
- Fallback rendering paths for different GPU capabilities

### Exception Hierarchy
```python
class SimulationError(Exception): pass
class PhysicsError(SimulationError): pass
class RenderingError(SimulationError): pass
class ShaderCompilationError(RenderingError): pass
class BufferAllocationError(RenderingError): pass
```

## Testing Strategy

### Unit Testing
- **Physics Module**: Test gravitational calculations, particle integration, relativistic effects
- **Math Utilities**: Verify vector operations, coordinate transformations
- **Configuration**: Validate parameter ranges and defaults

### Integration Testing  
- **GPU-CPU Data Transfer**: Verify particle data synchronization between CPU and GPU
- **Module Interfaces**: Test communication between display, physics, and rendering modules
- **Shader Programs**: Validate compute shader outputs against CPU calculations

### Performance Testing
- **Frame Rate Benchmarks**: Measure FPS with varying particle counts
- **Memory Usage**: Monitor GPU memory allocation and CPU memory usage
- **Compute Shader Performance**: Profile GPU compute times for different workloads

### Visual Testing
- **Reference Comparisons**: Compare outputs with known black hole simulation results
- **Physics Validation**: Verify orbital mechanics match theoretical predictions
- **Rendering Accuracy**: Check gravitational lensing effects against expected behavior

### Test Data
- Predefined particle configurations for consistent testing
- Known black hole parameters (mass, position) for reproducible results
- Benchmark scenarios with expected performance characteristics

## Documentation Strategy

### Module Documentation Requirements

Each module directory will contain a comprehensive `concepts.md` file that explains the theoretical foundations, implementation details, and visual diagrams relevant to that module's functionality.

**Display Module Documentation (`display/concepts.md`)**
- Window management and OpenGL context creation
- Input handling patterns and event processing
- UI framework integration (ImGui or similar)
- Camera mathematics and view transformations
- Mermaid diagrams showing input flow and rendering pipeline

**Physics Module Documentation (`physics/concepts.md`)**  
- General relativity concepts and Schwarzschild metric
- Gravitational force calculations and geodesic equations
- Numerical integration methods (RK4, Verlet, etc.)
- Event horizon and photon sphere physics
- Mathematical diagrams of spacetime curvature and particle trajectories
- Coordinate system transformations

**GPU Rendering Module Documentation (`gpu_rendering/concepts.md`)**
- OpenGL compute shader architecture and dispatch patterns
- Buffer management and GPU-CPU synchronization
- Shader compilation and program linking
- Vertex array objects and rendering pipelines
- Performance optimization techniques for GPU computing
- Memory layout diagrams and data flow charts

### Documentation Standards
- Each `concepts.md` file will include Mermaid diagrams for visual explanation
- Mathematical formulas will be rendered using LaTeX notation
- Code examples will demonstrate key concepts with practical implementations
- Cross-references between modules will be clearly documented
- Performance considerations and optimization strategies will be explained