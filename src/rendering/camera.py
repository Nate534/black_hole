import numpy as np
import pygame
from .camera_config import cdist, cspeed, czoom, cmove, crot

class Camera:
    def __init__(self, position=None):
        if position is None:
            self.position = np.array([0.0, cdist, -cdist], dtype=np.float64)
        else:
            self.position = np.array(position, dtype=np.float64)
            
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        
        # Camera controls
        self.move_speed = cmove
        self.zoom_speed = cspeed
        self.zoom = czoom
        self.rotation_speed = crot
        
        # For smooth movement
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = 5e9
        self.damping = 0.9
    
    def handle_event(self, event, dt):
        """Handle camera control events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset camera
                self.position = np.array([0.0, 1e12, -1e12], dtype=np.float64)
                self.zoom = 1.0
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                # Zoom in
                self.zoom *= 1.0 + self.zoom_speed
            elif event.key == pygame.K_MINUS:
                # Zoom out
                self.zoom /= 1.0 + self.zoom_speed
    
    def handle_continuous_movement(self, keys, dt):
        """Handle continuous key presses for camera movement"""
        move_vector = np.zeros(3, dtype=np.float64)
        
        if keys[pygame.K_w]:
            move_vector[2] += 1  # Forward 
            move_vector[2] -= 1  # Backward
        if keys[pygame.K_a]:
            move_vector[0] -= 1  # Left
        if keys[pygame.K_d]:
            move_vector[0] += 1  # Right
        if keys[pygame.K_q]:
            move_vector[1] += 1  # Up
        if keys[pygame.K_e]:
            move_vector[1] -= 1  # Down
        
        # Normalize if moving diagonally
        norm = np.linalg.norm(move_vector)
        if norm > 0:
            move_vector = move_vector / norm
            
        # Calculate forward and right vectors relative to camera orientation
        forward = normalize(self.target - self.position)
        right = normalize(np.cross(forward, self.up))
        
        # Transform movement to world space
        world_move = move_vector[0] * right + move_vector[1] * self.up + move_vector[2] * forward
        
        # Apply movement
        if np.any(world_move != 0):
            self.velocity += world_move * self.acceleration * dt
        else:
            # Apply damping when not moving
            self.velocity *= self.damping
            
        # Update position
        self.position += self.velocity * dt
        
        # Keep camera at a minimum distance from the black hole
        min_distance = 1e15
        distance_to_black_hole = np.linalg.norm(self.position)
        if distance_to_black_hole < min_distance:
            direction = normalize(self.position)
            self.position = direction * min_distance
            # Reflect velocity away from black hole
            self.velocity -= 2 * np.dot(self.velocity, direction) * direction
    
    def world_to_screen(self, world_pos, screen_width, screen_height):
        """Convert world coordinates to screen coordinates"""
        world_pos = np.array(world_pos, dtype=np.float64)
        
        # Calculate relative position to camera
        relative_pos = world_pos - self.position
        
        # Simple projection (assuming camera looks along forward vector)
        forward = normalize(self.target - self.position)
        
        # Project onto camera plane (simplified)
        if np.dot(relative_pos, forward) <= 0:
            return None  # Behind camera
        
        # Scale based on distance and zoom
        scale = 1e9 * self.zoom
        x = relative_pos[0] / scale
        y = relative_pos[1] / scale
        
        # Convert to screen coordinates
        screen_x = screen_width / 2 + x
        screen_y = screen_height / 2 - y
        
        return (int(screen_x), int(screen_y))

def normalize(vector):
    """Normalize a vector"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm