"""
Unit tests for Vector3 class.

Tests all mathematical operations, edge cases, and error conditions
for the Vector3 implementation.
"""

import unittest
import math
from physics.vector3 import Vector3


class TestVector3(unittest.TestCase):
    """Test cases for Vector3 class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.v1 = Vector3(1.0, 2.0, 3.0)
        self.v2 = Vector3(4.0, 5.0, 6.0)
        self.zero = Vector3.zero()
        self.unit_x = Vector3.unit_x()
        self.unit_y = Vector3.unit_y()
        self.unit_z = Vector3.unit_z()
    
    def test_initialization(self):
        """Test vector initialization."""
        v = Vector3(1.5, -2.5, 3.5)
        self.assertEqual(v.x, 1.5)
        self.assertEqual(v.y, -2.5)
        self.assertEqual(v.z, 3.5)
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        # Test known magnitude
        v = Vector3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.magnitude(), 5.0, places=10)
        
        # Test 3D magnitude
        v = Vector3(1.0, 2.0, 2.0)
        expected = math.sqrt(1 + 4 + 4)
        self.assertAlmostEqual(v.magnitude(), expected, places=10)
        
        # Test zero vector
        self.assertEqual(self.zero.magnitude(), 0.0)
        
        # Test unit vectors
        self.assertAlmostEqual(self.unit_x.magnitude(), 1.0, places=10)
        self.assertAlmostEqual(self.unit_y.magnitude(), 1.0, places=10)
        self.assertAlmostEqual(self.unit_z.magnitude(), 1.0, places=10)
    
    def test_magnitude_squared(self):
        """Test squared magnitude calculation."""
        v = Vector3(3.0, 4.0, 0.0)
        self.assertEqual(v.magnitude_squared(), 25.0)
        
        v = Vector3(1.0, 2.0, 2.0)
        self.assertEqual(v.magnitude_squared(), 9.0)
        
        self.assertEqual(self.zero.magnitude_squared(), 0.0)
    
    def test_normalize(self):
        """Test vector normalization."""
        # Test regular vector
        v = Vector3(3.0, 4.0, 0.0)
        normalized = v.normalize()
        self.assertAlmostEqual(normalized.magnitude(), 1.0, places=10)
        self.assertAlmostEqual(normalized.x, 0.6, places=10)
        self.assertAlmostEqual(normalized.y, 0.8, places=10)
        self.assertEqual(normalized.z, 0.0)
        
        # Test that original vector is unchanged
        self.assertEqual(v.x, 3.0)
        self.assertEqual(v.y, 4.0)
        self.assertEqual(v.z, 0.0)
        
        # Test unit vectors remain unit
        self.assertEqual(self.unit_x.normalize(), self.unit_x)
        
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector raises error."""
        with self.assertRaises(ValueError):
            self.zero.normalize()
    
    def test_dot_product(self):
        """Test dot product calculation."""
        # Test orthogonal vectors
        self.assertEqual(self.unit_x.dot(self.unit_y), 0.0)
        self.assertEqual(self.unit_y.dot(self.unit_z), 0.0)
        self.assertEqual(self.unit_z.dot(self.unit_x), 0.0)
        
        # Test parallel vectors
        self.assertEqual(self.unit_x.dot(self.unit_x), 1.0)
        
        # Test general case
        result = self.v1.dot(self.v2)  # (1,2,3) · (4,5,6) = 4+10+18 = 32
        self.assertEqual(result, 32.0)
        
        # Test with zero vector
        self.assertEqual(self.v1.dot(self.zero), 0.0)
    
    def test_cross_product(self):
        """Test cross product calculation."""
        # Test orthogonal unit vectors
        result = self.unit_x.cross(self.unit_y)
        self.assertEqual(result, self.unit_z)
        
        result = self.unit_y.cross(self.unit_z)
        self.assertEqual(result, self.unit_x)
        
        result = self.unit_z.cross(self.unit_x)
        self.assertEqual(result, self.unit_y)
        
        # Test anti-commutativity
        result1 = self.v1.cross(self.v2)
        result2 = self.v2.cross(self.v1)
        self.assertEqual(result1, -result2)
        
        # Test parallel vectors give zero
        result = self.unit_x.cross(self.unit_x)
        self.assertEqual(result, self.zero)
        
        # Test with zero vector
        result = self.v1.cross(self.zero)
        self.assertEqual(result, self.zero)
    
    def test_distance(self):
        """Test distance calculation."""
        # Distance between unit vectors
        dist = self.unit_x.distance_to(self.unit_y)
        expected = math.sqrt(2.0)  # sqrt((1-0)² + (0-1)² + (0-0)²)
        self.assertAlmostEqual(dist, expected, places=10)
        
        # Distance to self
        self.assertEqual(self.v1.distance_to(self.v1), 0.0)
        
        # Distance squared
        dist_sq = self.unit_x.distance_squared_to(self.unit_y)
        self.assertAlmostEqual(dist_sq, 2.0, places=10)
    
    def test_addition(self):
        """Test vector addition."""
        result = self.v1 + self.v2
        expected = Vector3(5.0, 7.0, 9.0)
        self.assertEqual(result, expected)
        
        # Addition with zero
        result = self.v1 + self.zero
        self.assertEqual(result, self.v1)
    
    def test_subtraction(self):
        """Test vector subtraction."""
        result = self.v2 - self.v1
        expected = Vector3(3.0, 3.0, 3.0)
        self.assertEqual(result, expected)
        
        # Subtraction from self
        result = self.v1 - self.v1
        self.assertEqual(result, self.zero)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        result = self.v1 * 2.0
        expected = Vector3(2.0, 4.0, 6.0)
        self.assertEqual(result, expected)
        
        # Reverse multiplication
        result = 2.0 * self.v1
        self.assertEqual(result, expected)
        
        # Multiplication by zero
        result = self.v1 * 0.0
        self.assertEqual(result, self.zero)
        
        # Multiplication by negative
        result = self.v1 * -1.0
        expected = Vector3(-1.0, -2.0, -3.0)
        self.assertEqual(result, expected)
    
    def test_scalar_division(self):
        """Test scalar division."""
        result = self.v1 / 2.0
        expected = Vector3(0.5, 1.0, 1.5)
        self.assertEqual(result, expected)
        
        # Division by one
        result = self.v1 / 1.0
        self.assertEqual(result, self.v1)
    
    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with self.assertRaises(ValueError):
            self.v1 / 0.0
    
    def test_negation(self):
        """Test vector negation."""
        result = -self.v1
        expected = Vector3(-1.0, -2.0, -3.0)
        self.assertEqual(result, expected)
        
        # Double negation
        result = -(-self.v1)
        self.assertEqual(result, self.v1)
    
    def test_equality(self):
        """Test vector equality."""
        v1_copy = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(self.v1, v1_copy)
        
        # Test inequality
        self.assertNotEqual(self.v1, self.v2)
        
        # Test floating point tolerance
        v_close = Vector3(1.0000000001, 2.0, 3.0)
        self.assertEqual(self.v1, v_close)
    
    def test_string_representation(self):
        """Test string representations."""
        str_repr = str(self.v1)
        self.assertIn("Vector3", str_repr)
        self.assertIn("1.000000", str_repr)
        
        repr_str = repr(self.v1)
        self.assertIn("Vector3", repr_str)
        self.assertIn("x=1.0", repr_str)
    
    def test_class_methods(self):
        """Test class method constructors."""
        # Test zero vector
        zero = Vector3.zero()
        self.assertEqual(zero.x, 0.0)
        self.assertEqual(zero.y, 0.0)
        self.assertEqual(zero.z, 0.0)
        
        # Test unit vectors
        unit_x = Vector3.unit_x()
        self.assertEqual(unit_x, Vector3(1.0, 0.0, 0.0))
        
        unit_y = Vector3.unit_y()
        self.assertEqual(unit_y, Vector3(0.0, 1.0, 0.0))
        
        unit_z = Vector3.unit_z()
        self.assertEqual(unit_z, Vector3(0.0, 0.0, 1.0))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very large numbers
        large_v = Vector3(1e10, 1e10, 1e10)
        self.assertAlmostEqual(large_v.magnitude(), math.sqrt(3) * 1e10, places=5)
        
        # Very small numbers
        small_v = Vector3(1e-10, 1e-10, 1e-10)
        self.assertAlmostEqual(small_v.magnitude(), math.sqrt(3) * 1e-10, places=15)
        
        # Mixed signs
        mixed_v = Vector3(-1.0, 2.0, -3.0)
        self.assertAlmostEqual(mixed_v.magnitude(), math.sqrt(14), places=10)


if __name__ == '__main__':
    unittest.main()