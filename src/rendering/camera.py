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
        
        self.move_speed = cmove
        self.zoom_speed = cspeed
        self.zoom = czoom
        self.rotation_speed = crot
        
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = 5e9
        self.damping = 0.9
    
    def handle_event(self, event, dt):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.position = np.array([0.0, 1e12, -1e12], dtype=np.float64)
                self.zoom = 1.0
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.zoom *= 1.0 + self.zoom_speed
            elif event.key == pygame.K_MINUS:
                self.zoom /= 1.0 + self.zoom_speed
    
    def handle_continuous_movement(self, keys, dt):
        move_vector = np.zeros(3, dtype=np.float64)
        
        if keys[pygame.K_w]:
            move_vector[2] += 1
            move_vector[2] -= 1
        if keys[pygame.K_a]:
            move_vector[0] -= 1
        if keys[pygame.K_d]:
            move_vector[0] += 1
        if keys[pygame.K_q]:
            move_vector[1] += 1
        if keys[pygame.K_e]:
            move_vector[1] -= 1
        
        norm = np.linalg.norm(move_vector)
        if norm > 0:
            move_vector = move_vector / norm
            
        forward = normalize(self.target - self.position)
        right = normalize(np.cross(forward, self.up))
        
        world_move = move_vector[0] * right + move_vector[1] * self.up + move_vector[2] * forward
        
        if np.any(world_move != 0):
            self.velocity += world_move * self.acceleration * dt
        else:
            self.velocity *= self.damping
            
        self.position += self.velocity * dt
        
        min_distance = 1e15
        distance_to_black_hole = np.linalg.norm(self.position)
        if distance_to_black_hole < min_distance:
            direction = normalize(self.position)
            self.position = direction * min_distance
            self.velocity -= 2 * np.dot(self.velocity, direction) * direction
    
    def world_to_screen(self, world_pos, screen_width, screen_height):
        world_pos = np.array(world_pos, dtype=np.float64)
        
        relative_pos = world_pos - self.position
        
        forward = normalize(self.target - self.position)
        
        if np.dot(relative_pos, forward) <= 0:
            return None
        
        scale = 1e9 * self.zoom
        x = relative_pos[0] / scale
        y = relative_pos[1] / scale
        
        screen_x = screen_width / 2 + x
        screen_y = screen_height / 2 - y
        
        return (int(screen_x), int(screen_y))

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm