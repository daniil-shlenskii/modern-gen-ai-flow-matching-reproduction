import numpy as np
from numpy import sin


def generate_checkerboard_dataset(num_points, bound, num_cells):
    """
    Generates a checkerboard.

    Example usage::

        >>> import matplotlib.pyplot as plt
        >>> x = generate_checkerboard_dataset(num_points=1000, bound=10, num_cells=4)
        >>> plt.scatter(x[:,0], x[:,1], s=1)
        >>> plt.show()
    
    Parameters
    ----------
        num_points : int - Number of points to generate.
        cbound : int - The bound of points' coordinates, coordinates of points are within (-bound, bound).
        num_cells : int - Number of cells in the image. Example: num_cells = 4 will generate a checkerboard with 4x4 cells.

    Returns
    -------
    X : np.ndarray - An array of shape (num_points, 2), ponts with 2d coordinates.
    """
    coef = bound / ((num_cells // 2) * np.pi)

    x = np.random.uniform(-bound, bound, size=(num_points, 2))

    mask = np.logical_or(
        np.logical_and(sin(x[:,0] / coef) > 0.0, sin(x[:,1] / coef) > 0.0),
        np.logical_and(sin(x[:,0] / coef) < 0.0, sin(x[:,1] / coef) < 0.0)
    )
    res = x[mask]
    y = x[~mask]
    y[:,0] *= -1
    res = np.vstack((res, y))
    return res
