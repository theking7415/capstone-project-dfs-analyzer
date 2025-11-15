"""
Statistical analysis utilities for DFS experiments.

Provides functions for analyzing discovery number distributions
and validating the (n-1)/2 conjecture.
"""

from typing import Any

import numpy as np


def compute_overall_average(summary_stats: dict[Any, Any]) -> float:
    """
    Computes the overall average discovery number across all vertices.

    Args:
        summary_stats: Maps vertices to scipy.stats.DescribeResult objects.

    Returns:
        Mean of all vertex means.
    """
    # Extracts mean values from all vertices
    all_means = [stat.mean for stat in summary_stats.values()]
    # Returns average of all means
    return np.mean(all_means)


def compute_overall_std(summary_stats: dict[Any, Any]) -> float:
    """
    Computes standard deviation of mean discovery numbers across vertices.

    Args:
        summary_stats: Maps vertices to scipy.stats.DescribeResult objects.

    Returns:
        Standard deviation of vertex means.
    """
    # Extracts mean values from all vertices
    all_means = [stat.mean for stat in summary_stats.values()]
    # Returns standard deviation of means
    return np.std(all_means)


def validate_conjecture(
    num_vertices: int, observed_mean: float, tolerance: float = 0.01
) -> dict[str, Any]:
    """
    Validates the (n-1)/2 conjecture for a graph.

    Compares observed mean against theoretical (n-1)/2 value.
    Determines if conjecture holds within tolerance.

    Args:
        num_vertices: Specifies total number of vertices in graph.
        observed_mean: Specifies observed average discovery number.
        tolerance: Sets relative error tolerance for validation.

    Returns:
        Dictionary containing validation metrics and results.
    """
    # Calculates theoretical prediction
    theoretical_value = (num_vertices - 1) / 2
    # Computes absolute difference
    absolute_error = abs(observed_mean - theoretical_value)
    # Calculates relative error as fraction of theoretical value
    relative_error = absolute_error / theoretical_value if theoretical_value != 0 else 0
    # Checks if error is within tolerance
    is_valid = relative_error < tolerance

    return {
        "theoretical_value": theoretical_value,
        "observed_value": observed_mean,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "is_valid": is_valid,
        "tolerance": tolerance,
    }


def format_validation_result(validation: dict[str, Any]) -> str:
    """
    Formats validation results as human-readable string.

    Args:
        validation: Contains dictionary returned by validate_conjecture().

    Returns:
        Formatted string with validation details.
    """
    # Determines status symbol based on validation result
    status = "✓ VALID" if validation["is_valid"] else "✗ INVALID"
    # Builds formatted output lines
    lines = [
        f"Theoretical (n-1)/2: {validation['theoretical_value']:.4f}",
        f"Observed mean: {validation['observed_value']:.4f}",
        f"Absolute error: {validation['absolute_error']:.6f}",
        f"Relative error: {validation['relative_error']*100:.4f}%",
        f"Status: {status} (tolerance: {validation['tolerance']*100:.2f}%)",
    ]
    return "\n".join(lines)


def compute_percentiles(values: list[float], percentiles: list[float]) -> dict[float, float]:
    """
    Computes percentiles for a list of values.

    Args:
        values: Contains list of numerical values.
        percentiles: Specifies list of percentiles to compute.

    Returns:
        Dictionary mapping percentile to its value.
    """
    # Computes each requested percentile
    return {p: np.percentile(values, p) for p in percentiles}


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """
    Computes summary statistics for a distribution.

    Args:
        values: Contains list of numerical values.

    Returns:
        Dictionary with mean, median, std, min, max, and quartiles.
    """
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
    }
