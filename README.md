# Black Hole

A C# application for physics simulation and visualization.

## Project Description

Black Hole is a .NET-based project that combines physics simulations with rendering capabilities. The project is organized into modular components for physics calculations, rendering, and utility functions.

## Setup/Installation

### Prerequisites
- .NET 8.0 SDK or later
- OpenTK libraries (automatically included via NuGet)

### Installation Steps

1. Clone or download the project
2. Navigate to the project directory:
   ```bash
   cd black_hole
   ```
3. Restore dependencies:
   ```bash
   dotnet restore
   ```
4. Build the project:
   ```bash
   dotnet build
   ```
5. Run the program:
   ```bash
   dotnet run
   ```

## Dependencies

- **OpenTK** (4.9.4) - Open Toolkit for graphics rendering and window management
- **OpenTK.Mathematics** (4.9.4) - Mathematical utilities for vector and matrix operations
- **.NET 8.0** - Target framework

## Project Structure

- `physics/` - Physics simulation logic
- `rendering/` - Graphics rendering components
- `utils/` - Utility functions and helpers
- `Objects/` - Object definitions
- `Program.cs` - Entry point
