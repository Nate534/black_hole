import numpy as np

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def rotate_vector(vector, axis, angle):
    axis = normalize(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    return (vector * cos_angle + 
            np.cross(axis, vector) * sin_angle + 
            axis * np.dot(axis, vector) * (1 - cos_angle))