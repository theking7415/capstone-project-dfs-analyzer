#!/usr/bin/env python3
"""
Creates separate formulas for Low-D (3-5) and Mid-High-D (6-13) hypercubes
"""

import numpy as np

def hypercube_coefficients_lowD(d):
    """
    For small dimensions 3D-5D only
    These have tiny coefficients - use actual values from experiments
    """
    # Direct lookup table for 3D, 4D, 5D (from high-quality experiments)
    lookup = {
        3: (0.009724, 0.052608, 0.027811, 0.196958),  # R²=1.0
        4: (-0.060100, 0.365750, -0.169250, -0.192100),  # R²=1.0
        5: (0.013292, -0.251304, 1.672905, -2.216440)   # R²=0.998
    }
    
    if d in lookup:
        return lookup[d]
    else:
        # Linear interpolation for fractional dimensions (if needed)
        print(f"Warning: Dimension {d} not in low-D range (3-5)")
        return None

def hypercube_coefficients_midHighD(d):
    """
    For dimensions 6D-13D
    Uses fitted formulas with R² ≈ 0.99
    """
    a = 8.4426998447e-05*d**3 + 4.1180329090e-03*d**2 + -3.1303810812e-02*d + 0.0422080393
    b = -1.8513742162e-02*d**3 + 2.4772916178e-01*d**2 + -1.3131378466e+00*d + 2.4658320894
    c = 3.6969631023e-01*np.exp(0.4387811408*d) + -1.9103070596
    d_coeff = -6.8304601244e-01*d**3 + 1.1914160724e+01*d**2 + -6.8263451054e+01*d + 121.3491868581
    return a, b, c, d_coeff

def hypercube_coefficients(d):
    """
    Unified hypercube formula - automatically selects the right range
    
    Valid ranges:
    - 3D-5D: Direct lookup (perfect accuracy)
    - 6D-13D: Fitted formulas (R² ≈ 0.99)
    
    Usage:
        a, b, c, d = hypercube_coefficients(7)
        deviation(layer) = a*layer³ + b*layer² + c*layer + d  
        mean_layer = (n-1)/2 + deviation(layer)
    """
    if 3 <= d <= 5:
        return hypercube_coefficients_lowD(d)
    elif 6 <= d <= 13:
        return hypercube_coefficients_midHighD(d)
    else:
        print(f"Warning: Dimension {d} outside validated range (3-13)")
        print(f"Using mid-high-D formula (may be inaccurate)")
        return hypercube_coefficients_midHighD(d)

if __name__ == '__main__':
    print("HYPERCUBE FORMULA - Split Range Approach")
    print("="*60)
    
    # Test all dimensions
    for d in range(3, 14):
        a, b, c, d_coeff = hypercube_coefficients(d)
        print(f"\n{d}D: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d_coeff:.6f}")
        
        # Show prediction for layer 3 as example
        n = 2**d
        expected = (n-1)/2
        deviation_layer3 = a*3**3 + b*3**2 + c*3 + d_coeff
        mean_layer3 = expected + deviation_layer3
        print(f"     Layer 3 prediction: {mean_layer3:.2f} (expected: {expected:.2f}, deviation: {deviation_layer3:+.2f})")
