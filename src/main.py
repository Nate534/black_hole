import pygame
from pygame.locals import *
import numpy as np
import sys
from physics.black_hole import BlackHole
from physics.particle import ParticleSystem
from rendering.renderer import Renderer
from rendering.camera import Camera
from physics.constants import G, C

def main():
    pygame.init()
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Black Hole Simulation")
    
    # Create black hole with larger mass for more dramatic effects
    black_hole = BlackHole(mass=4e37, position=(0, 0, 0))
    particle_system = ParticleSystem()
    
    # Create camera with better initial position
    camera = Camera(position=(0, 2e11, -3e11))
    renderer = Renderer(width, height, screen)
    
    print(f"Black hole mass: {black_hole.mass:.2e} kg")
    print(f"Schwarzschild radius: {black_hole.schwarz_radius:.2e} m")
    
    # Create accretion disk with varying properties
    num_particles = 300
    inner_radius = black_hole.schwarz_radius * 2.5
    outer_radius = black_hole.schwarz_radius * 12
    
    for i in range(num_particles):
        # Vary distance from black hole
        distance = np.random.uniform(inner_radius, outer_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Position in the accretion disk plane
        x = distance * np.cos(angle)
        z = distance * np.sin(angle)
        y = np.random.uniform(-outer_radius/50, outer_radius/50)  # Slight thickness
        
        # Calculate orbital velocity with some random perturbation
        orbital_speed = np.sqrt(G * black_hole.mass / distance)
        perturbation = np.random.uniform(0.95, 1.05)  # Small randomness
        vx = -orbital_speed * np.sin(angle) * perturbation
        vz = orbital_speed * np.cos(angle) * perturbation
        vy = np.random.uniform(-orbital_speed/20, orbital_speed/20)
        
        # Color based on distance from black hole (redshift effect)
        distance_ratio = (distance - inner_radius) / (outer_radius - inner_radius)
        # Hotter (blue) near the center, cooler (red) further out
        r = min(1.0, 0.2 + distance_ratio * 0.8)
        g = max(0.0, 0.5 - distance_ratio * 0.5)
        b = max(0.0, 1.0 - distance_ratio * 0.8)
        
        particle_system.add_particle(
            mass=np.random.uniform(1e9, 1e10),
            position=(x, y, z),
            velocity=(vx, vy, vz),
            colour=(r, g, b)
        )
    
    print(f"Added {len(particle_system.particles)} particles")
    
    # Add some particles with high velocity for jet effects
    for i in range(30):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = inner_radius * 1.2
        x = distance * np.cos(angle)
        z = distance * np.sin(angle)
        y_sign = 1 if np.random.random() > 0.5 else -1
        y = y_sign * distance * 0.3
        
        # High velocity in y-direction for jet effect
        vx = np.random.uniform(-2e7, 2e7)
        vz = np.random.uniform(-2e7, 2e7)
        vy = np.random.uniform(8e7, 2e8) * y_sign
        
        particle_system.add_particle(
            mass=np.random.uniform(1e8, 1e9),
            position=(x, y, z),
            velocity=(vx, vy, vz),
            colour=(0.7, 0.7, 1.0)  # Bluish color for jets
        )
    
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    time_step = 50.0  # Reduced time step for better stability
    
    # Main simulation loop
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    particle_system.grav_enabled = not particle_system.grav_enabled
                    print(f"Gravity {'enabled' if particle_system.grav_enabled else 'disabled'}")
                elif event.key == pygame.K_r:
                    # Reset camera
                    camera.position = np.array([0.0, 2e11, -3e11], dtype=np.float64)
                    camera.zoom = 1.0
                    print("Camera reset")
            
            # Handle camera controls
            camera.handle_event(event, dt)
        
        # Handle continuous key presses for smoother camera movement
        keys = pygame.key.get_pressed()
        camera.handle_continuous_movement(keys, dt)
        
        # Clear screen
        screen.fill((0, 0, 0))  # Black background
        
        # Update physics
        particle_system.update(black_hole, time_step)
        
        # add new particles
        if frame_count % 120 == 0 and len(particle_system.particles) < num_particles * 0.9:
            # Add a new particle to the accretion disk
            distance = np.random.uniform(inner_radius * 1.5, outer_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            x = distance * np.cos(angle)
            z = distance * np.sin(angle)
            y = np.random.uniform(-outer_radius/50, outer_radius/50)
            
            orbital_speed = np.sqrt(G * black_hole.mass / distance)
            vx = -orbital_speed * np.sin(angle)
            vz = orbital_speed * np.cos(angle)
            
            distance_ratio = (distance - inner_radius) / (outer_radius - inner_radius)
            r = min(1.0, 0.2 + distance_ratio * 0.8)
            g = max(0.0, 0.5 - distance_ratio * 0.5)
            b = max(0.0, 1.0 - distance_ratio * 0.8)
            
            particle_system.add_particle(
                mass=np.random.uniform(1e9, 1e10),
                position=(x, y, z),
                velocity=(vx, 0, vz),
                colour=(r, g, b)
            )
        
        # Render
        try:
            renderer.render(black_hole, particle_system, camera)
        except Exception as e:
            print(f"Rendering error: {e}")
            # Fallback rendering
            pygame.draw.circle(screen, (255, 0, 0), (width//2, height//2), 50)
        
        pygame.display.flip()
        frame_count += 1
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()