# Requirements Document

## Introduction

This project aims to create a black hole simulation using Python and OpenGL, inspired by the referenced C++ implementation. The simulation will demonstrate gravitational lensing effects, particle dynamics around black holes, and real-time visualization. The architecture will be modular with separate components for display, physics calculations, and GPU rendering, all running within a Python virtual environment.

## Requirements

### Requirement 1

**User Story:** As a physics enthusiast, I want to visualize a black hole's gravitational effects on light and particles, so that I can understand how spacetime curvature affects the behavior of matter and energy.

#### Acceptance Criteria

1. WHEN the simulation starts THEN the system SHALL display a black hole at the center of the viewport
2. WHEN light rays or particles approach the black hole THEN the system SHALL demonstrate gravitational lensing effects
3. WHEN particles orbit the black hole THEN the system SHALL show realistic orbital mechanics based on general relativity
4. IF a particle crosses the event horizon THEN the system SHALL handle the particle appropriately (disappear or show spaghettification)

### Requirement 2

**User Story:** As a developer, I want the codebase to have clear separation of concerns between display, physics, and rendering, so that I can maintain and extend each component independently.

#### Acceptance Criteria

1. WHEN examining the project structure THEN the system SHALL have three distinct modules: display, physics, and gpu_rendering
2. WHEN modifying physics calculations THEN the system SHALL not require changes to display or rendering code
3. WHEN updating rendering techniques THEN the system SHALL not affect physics or display logic
4. IF new display features are added THEN the system SHALL integrate through well-defined interfaces

### Requirement 3

**User Story:** As a user, I want the simulation to run smoothly in real-time, so that I can interactively explore black hole physics without lag or stuttering.

#### Acceptance Criteria

1. WHEN the simulation is running THEN the system SHALL maintain at least 30 FPS during normal operation
2. WHEN calculating particle trajectories THEN the system SHALL use GPU acceleration for performance
3. WHEN rendering complex scenes THEN the system SHALL optimize OpenGL calls for smooth playback
4. IF the system detects performance issues THEN the system SHALL provide options to reduce quality for better performance

### Requirement 4

**User Story:** As a developer, I want the project to run in an isolated Python virtual environment, so that I can manage dependencies without conflicts and ensure reproducible builds.

#### Acceptance Criteria

1. WHEN setting up the project THEN the system SHALL create and activate a Python virtual environment
2. WHEN installing dependencies THEN the system SHALL use pip with a requirements.txt file
3. WHEN running the simulation THEN the system SHALL execute within the virtual environment
4. IF dependencies conflict THEN the system SHALL isolate them within the virtual environment

### Requirement 5

**User Story:** As a physics student, I want to see accurate black hole physics including event horizons and Schwarzschild radius effects, so that I can learn about general relativity concepts.

#### Acceptance Criteria

1. WHEN a black hole is created THEN the system SHALL calculate and display the event horizon based on mass
2. WHEN light approaches the photon sphere THEN the system SHALL show unstable circular orbits
3. WHEN particles have different initial velocities THEN the system SHALL demonstrate escape velocity concepts
4. IF particles approach the singularity THEN the system SHALL show increasing gravitational effects

### Requirement 6

**User Story:** As a user, I want interactive controls to adjust simulation parameters, so that I can experiment with different black hole masses, particle properties, and viewing angles.

#### Acceptance Criteria

1. WHEN using the interface THEN the system SHALL provide controls for black hole mass adjustment
2. WHEN spawning particles THEN the system SHALL allow setting initial position, velocity, and mass
3. WHEN viewing the simulation THEN the system SHALL support camera rotation and zoom
4. IF parameters are changed during runtime THEN the system SHALL update the simulation in real-time

### Requirement 7

**User Story:** As a developer, I want clear class structures for black holes and particles, so that I can easily understand and extend the physics implementation.

#### Acceptance Criteria

1. WHEN examining the physics module THEN the system SHALL have a dedicated BlackHole class
2. WHEN working with particles THEN the system SHALL have a Particle class with position, velocity, and mass properties
3. WHEN calculating interactions THEN the system SHALL use well-defined methods for gravitational effects
4. IF new particle types are needed THEN the system SHALL support extension through inheritance or composition