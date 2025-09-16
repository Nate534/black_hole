import pygame
import numpy as np

class Renderer:
    def __init__(self, width, height, screen):
        self.width = width
        self.height = height
        self.screen = screen
        self.font = pygame.font.Font(None, 24)
        
        # Create surface for glow effects
        self.glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Precompute accretion disk texture
        self.disk_texture = self.create_disk_texture(256)
        
    def create_disk_texture(self, size):
        """Create a texture for the accretion disk"""
        texture = pygame.Surface((size, size), pygame.SRCALPHA)
        
        center = size // 2
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                distance = np.sqrt(dx*dx + dy*dy) / center
                
                if distance < 1.0:
                    # Color based on distance (red to blue)
                    r = min(255, int(100 + 155 * (1 - distance)))
                    g = max(0, int(50 + 100 * (1 - distance)))
                    b = max(0, int(50 + 205 * distance))
                    a = min(255, int(200 * (1 - distance**2)))
                    
                    texture.set_at((x, y), (r, g, b, a))
        
        return texture
    
    def render(self, black_hole, particle_system, camera):
        """Render the black hole simulation"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        self.glow_surface.fill((0, 0, 0, 0))
        
        # Render black hole with accretion disk
        self.render_black_hole(black_hole, camera)
        
        # Render particles
        self.render_particles(particle_system, camera)
        
        # Apply glow effect
        self.apply_glow()
        
        # Render UI
        self.render_ui(camera, particle_system, black_hole)
    
    def render_black_hole(self, black_hole, camera):
        """Render the black hole with detailed accretion disk"""
        # Convert black hole position to screen coordinates
        screen_pos = camera.world_to_screen(black_hole.position, self.width, self.height)
        
        if screen_pos:
            # Calculate sizes based on Schwarzschild radius and zoom
            schwarz_radius_px = max(5, int(black_hole.schwarz_radius / (1e9 * camera.zoom)))
            disk_radius_px = schwarz_radius_px * 6
            
            # Draw accretion disk
            if disk_radius_px > 5:
                # Create a scaled version of the disk texture
                scaled_disk = pygame.transform.scale(
                    self.disk_texture, 
                    (disk_radius_px * 2, disk_radius_px * 2)
                )
                disk_rect = scaled_disk.get_rect(center=screen_pos)
                self.screen.blit(scaled_disk, disk_rect)
            
            # Draw photon sphere (just inside the event horizon)
            photon_radius = int(schwarz_radius_px * 1.5)
            if photon_radius > 2:
                pygame.draw.circle(self.glow_surface, (150, 150, 255, 100), screen_pos, photon_radius, 2)
            
            # Draw event horizon
            if schwarz_radius_px > 2:
                # Draw a dark circle for the event horizon
                pygame.draw.circle(self.screen, (10, 10, 10), screen_pos, schwarz_radius_px)
                
                # Add a slight glow around the event horizon
                pygame.draw.circle(self.glow_surface, (50, 50, 150, 50), screen_pos, schwarz_radius_px + 2)
            
            # Draw black hole shadow (larger than event horizon)
            shadow_radius = int(schwarz_radius_px * 2.5)
            if shadow_radius > 5:
                pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, shadow_radius)
                
                # Add gravitational lensing effect (distortion ring)
                for i in range(3):
                    ring_radius = shadow_radius + 5 + i * 3
                    pygame.draw.circle(self.glow_surface, (100, 100, 200, 30), screen_pos, ring_radius, 1)
    
    def render_particles(self, particle_system, camera):
        """Render all particles with glow effects"""
        for particle in particle_system.particles:
            screen_pos = camera.world_to_screen(particle.position, self.width, self.height)
            
            if screen_pos:
                # Convert color from (0,1) range to (0,255) range
                color = (
                    int(particle.colour[0] * 255),
                    int(particle.colour[1] * 255),
                    int(particle.colour[2] * 255)
                )
                
                # Calculate particle size based on mass
                particle_size = max(1, int(2 + np.log10(particle.mass) / 2))
                
                # Draw glow effect
                glow_radius = particle_size * 3
                glow_color = (color[0], color[1], color[2], 100)
                pygame.draw.circle(self.glow_surface, glow_color, screen_pos, glow_radius)
                
                # Draw particle
                pygame.draw.circle(self.screen, color, screen_pos, particle_size)
                
                # Draw trail if it exists
                if len(particle.trail) > 1:
                    self.render_trail(particle, camera)
    
    def render_trail(self, particle, camera):
        """Render particle trail"""
        trail_points = []
        for i, trail_pos in enumerate(particle.trail[-15:]):  # Last 15 trail points
            trail_screen_pos = camera.world_to_screen(trail_pos, self.width, self.height)
            if trail_screen_pos:
                trail_points.append(trail_screen_pos)
        
        if len(trail_points) > 1:
            # Draw trail as connected lines with fading effect
            for i in range(len(trail_points) - 1):
                alpha = int(200 * (i + 1) / len(trail_points))  # Fade from transparent to opaque
                trail_color = (
                    int(particle.colour[0] * 255),
                    int(particle.colour[1] * 255),
                    int(particle.colour[2] * 255),
                    alpha
                )
                pygame.draw.line(self.glow_surface, trail_color, trail_points[i], trail_points[i + 1], 2)
    
    def apply_glow(self):
        """Apply glow effect by blurring the glow surface and combining with main screen"""
        # Simple blur effect by scaling down and up
        small = pygame.transform.smoothscale(self.glow_surface, (self.width//4, self.height//4))
        blurred = pygame.transform.smoothscale(small, (self.width, self.height))
        self.screen.blit(blurred, (0, 0), special_flags=pygame.BLEND_ADD)
    
    def render_ui(self, camera, particle_system, black_hole):
        """Render UI information"""
        # Camera info
        cam_text = f"Camera: ({camera.position[0]:.2e}, {camera.position[1]:.2e}, {camera.position[2]:.2e})"
        cam_surface = self.font.render(cam_text, True, (255, 255, 255))
        self.screen.blit(cam_surface, (10, 10))
        
        # Zoom info
        zoom_text = f"Zoom: {camera.zoom:.2f}"
        zoom_surface = self.font.render(zoom_text, True, (255, 255, 255))
        self.screen.blit(zoom_surface, (10, 40))
        
        # Particle count
        particle_text = f"Particles: {len(particle_system.particles)}"
        particle_surface = self.font.render(particle_text, True, (255, 255, 255))
        self.screen.blit(particle_surface, (10, 70))
        
        # Gravity status
        gravity_text = f"Gravity: {'ON' if particle_system.grav_enabled else 'OFF'} (G to toggle)"
        gravity_surface = self.font.render(gravity_text, True, (255, 255, 255))
        self.screen.blit(gravity_surface, (10, 100))
        
        # Black hole info
        bh_text = f"BH Mass: {black_hole.mass:.2e} kg"
        bh_surface = self.font.render(bh_text, True, (255, 255, 255))
        self.screen.blit(bh_surface, (10, 130))
        
        rs_text = f"Schwarzschild Radius: {black_hole.schwarz_radius:.2e} m"
        rs_surface = self.font.render(rs_text, True, (255, 255, 255))
        self.screen.blit(rs_surface, (10, 160))
        
        # Controls
        controls = [
            "Controls:",
            "WASD - Move camera",
            "QE - Up/Down",
            "R - Reset camera",
            "+/- - Zoom",
            "G - Toggle gravity",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            control_surface = self.font.render(control, True, (200, 200, 200))
            self.screen.blit(control_surface, (self.width - 200, 10 + i * 25))