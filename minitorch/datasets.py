import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """
    Generate N random 2D points in the unit square [0, 1] x [0, 1].

    Args:
        N: Number of points to generate

    Returns:
        List of N tuples, each containing two float coordinates (x_1, x_2)
        where 0 <= x_1, x_2 < 1
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    A dataset for 2D point classification.

    Attributes:
        N: Number of data points in the dataset
        X: List of 2D points, where each point is a tuple (x_1, x_2)
        y: List of binary labels (0 or 1) corresponding to each point
    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generate a simple vertical split dataset.

    Points are classified based on whether they are on the left or right half
    of the unit square. Points with x_1 < 0.5 are labeled as class 1,
    and points with x_1 >= 0.5 are labeled as class 0.

    This creates a linearly separable dataset with a vertical decision boundary
    at x_1 = 0.5.

    Args:
        N: Number of random points to generate

    Returns:
        Graph containing N points and their binary classifications
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generate a diagonal split dataset.

    Points are classified based on whether they are below or above a diagonal
    line. Points where x_1 + x_2 < 0.5 are labeled as class 1, and points
    where x_1 + x_2 >= 0.5 are labeled as class 0.

    This creates a linearly separable dataset with a diagonal decision boundary
    from (0.5, 0) to (0, 0.5).

    Args:
        N: Number of random points to generate

    Returns:
        Graph containing N points and their binary classifications
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generate a dataset with two vertical strips.

    Points are classified as class 1 if they are in the left strip (x_1 < 0.2)
    or right strip (x_1 > 0.8), and class 0 if they are in the middle region
    (0.2 <= x_1 <= 0.8).

    This creates a non-linearly separable dataset that requires at least two
    decision boundaries to classify correctly.

    Args:
        N: Number of random points to generate

    Returns:
        Graph containing N points and their binary classifications
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generate an XOR (exclusive or) dataset.

    Points are classified as class 1 if they are in the top-left quadrant
    (x_1 < 0.5 and x_2 > 0.5) or bottom-right quadrant (x_1 > 0.5 and x_2 < 0.5).
    Points in the top-right or bottom-left quadrants are labeled as class 0.

    This is a classic non-linearly separable dataset that cannot be solved
    with a single linear classifier, demonstrating the need for non-linear
    activation functions or multiple layers in neural networks.

    Args:
        N: Number of random points to generate

    Returns:
        Graph containing N points and their binary classifications
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generate a circular/annular dataset.

    Points are classified based on their distance from the center (0.5, 0.5).
    Points outside a circle of radius sqrt(0.1) centered at (0.5, 0.5) are
    labeled as class 1, while points inside the circle are labeled as class 0.

    This creates a non-linearly separable dataset with a circular decision
    boundary, requiring non-linear transformations to classify correctly.

    Args:
        N: Number of random points to generate

    Returns:
        Graph containing N points and their binary classifications
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generate a two-spiral dataset.

    Creates two interleaving spiral patterns, one for each class. The first
    spiral (class 0) and second spiral (class 1) wind around each other,
    creating a highly non-linear classification problem.

    This is one of the most challenging 2D classification datasets, requiring
    sophisticated non-linear decision boundaries to separate the two spirals.
    It's commonly used to test the capacity of neural networks to learn
    complex patterns.

    Args:
        N: Number of points to generate (should be even for balanced classes)

    Returns:
        Graph containing N points arranged in two spirals with their binary
        classifications
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
