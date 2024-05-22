from numpy import array as np_array, absolute, arctan, power as np_power, sqrt as np_sqrt
from cv2 import filter2D # pylint: disable=E0611
import numpy as np


PREWITT_KERNEL_X: np_array = np_array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


PREWITT_KERNEL_Y: np_array = np_array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])


SOBEL_KERNEL_X: np_array = np_array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])


SOBEL_KERNEL_Y: np_array = np_array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


SCHARR_KERNEL_X: np_array = np_array([
    [3, 0, -3],
    [10, 0, -10],
    [3, 0, -3]
])


SCHARR_KERNEL_Y: np_array = np_array([
    [3, 10, 3],
    [0, 0, 0],
    [-3, -10, -3]
])


__all__ = ['gabor_kernel']


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)


def gabor_kernel(
    frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
    n_stds=3, offset=0, dtype=np.complex128
):
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    if np.dtype(dtype).kind != 'c':
        raise ValueError("dtype must be complex")

    ct = np.cos(theta)
    st = np.sin(theta)
    x0 = np.ceil(
        max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1)
    )
    y0 = np.ceil(
        max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1)
    )
    y, x = np.meshgrid(np.arange(-y0, y0 + 1),
                       np.arange(-x0, x0 + 1),
                       indexing='ij',
                       sparse=True)
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    g = np.empty(roty.shape, dtype=dtype)
    np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2)
           + 1j * (2 * np.pi * frequency * rotx + offset),
           out=g)
    g *= 1 / (2 * np.pi * sigma_x * sigma_y)

    return g


def gradient_x(image: np_array, kernel: np_array = PREWITT_KERNEL_X) -> np_array:
    return filter2D(image, -1, kernel)


def gradient_y(image: np_array, kernel: np_array = PREWITT_KERNEL_Y) -> np_array:
    return filter2D(image, -1, kernel)


def gradient(
    image: np_array, kernel_x: np_array = PREWITT_KERNEL_X,
    kernel_y: np_array = PREWITT_KERNEL_Y
) -> np_array:

    # Compute the gradient of x and y
    gradient_x_: np_array = gradient_x(image, kernel_x)
    gradient_y_: np_array = gradient_y(image, kernel_y)

    # Compute the absolute values of gradient
    gradient_x_abs: np_array = absolute(gradient_x_)
    gradient_y_abs: np_array = absolute(gradient_y_)

    return gradient_x_abs + gradient_y_abs


def gradient_direction(
    image: np_array, kernel_x: np_array = PREWITT_KERNEL_X,
    kernel_y: np_array = PREWITT_KERNEL_Y
) -> np_array:
    # Compute the gradient of x and y
    gradient_x_: np_array = gradient_x(image, kernel_x)
    gradient_y_: np_array = gradient_y(image, kernel_y)

    return arctan(gradient_y_ / gradient_x_)


def gradient_magnitude(
    image: np_array, kernel_x: np_array = PREWITT_KERNEL_X,
    kernel_y: np_array = PREWITT_KERNEL_Y
) -> np_array:
    # Compute the gradient of x and y
    gradient_x_: np_array = gradient_x(image, kernel_x)
    gradient_y_: np_array = gradient_y(image, kernel_y)

    # Compute the square of the gradient
    gradient_x_square: np_array = np_power(gradient_x_, 2)
    gradient_y_square: np_array = np_power(gradient_y_, 2)

    return np_sqrt(gradient_x_square + gradient_y_square)
