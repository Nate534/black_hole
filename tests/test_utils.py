import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.vectors import normalize, rotate_vector

class TestVectors:
    def test_normalize(self):
        vec = np.array([3, 4, 0])
        norm_vec = normalize(vec)
        assert np.allclose(np.linalg.norm(norm_vec), 1.0)
        assert np.allclose(norm_vec, (0.6, 0.8, 0))

    def test_normalize_zero(self):
        vec = np.array([0, 0, 0])
        norm_vec = normalize(vec)
        assert np.array_equal(norm_vec, (0, 0, 0))

    def test_rotate_vector_x_axis(self):
        vec = np.array([0, 0, 1])
        axis = np.array([1, 0, 0])
        rotated = rotate_vector(vec, axis, np.pi / 2)
        expected = np.array([0, -1, 0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_rotate_vector_identity(self):
        vec = np.array([1, 0, 0])
        axis = np.array([0, 1, 0])
        rotated = rotate_vector(vec, axis, 0)
        assert np.allclose(rotated, vec)