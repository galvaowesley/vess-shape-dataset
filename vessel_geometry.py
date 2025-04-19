
import math
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import line


def create_endpoints(img_shape, min_dist, max_dist, padding):
    """Create two random points inside the region defined by `img_shape`.

    Args:
        img_shape (tuple): Number of rows and columns of the image
        min_dist (float): Minimum distance between the points
        max_dist (float): Maximum distance between the points
        padding (int): The points have a minimum distance equal to
        `padding` from the border of the image

    Returns:
        points: The generated points
        distance: The distance between the points
    """

    valid = False
    nr, nc = img_shape
    while not valid:
        # Random selection of points
        p1r = np.random.randint(padding, nr-padding)
        p1c = np.random.randint(padding, nc-padding)
        p2r = np.random.randint(padding, nr-padding)
        p2c = np.random.randint(padding, nc-padding)

        # Euclidean distance between the points
        distance = np.sqrt((p1r - p2r) ** 2 + (p1c - p2c) ** 2)

        if min_dist < distance < max_dist:
            valid = True

    points = np.array([(p1c, p1r), (p2c, p2r)])

    return points, distance


def bezier(points, precision):
    """Create Bezier curve.

    Args:
        points (list): List containing the control points of the curve
        precision (int): Number of points to use to represent the cruve

    Returns:
        curve: Numpy array containing the generated curve
    """

    ts = np.linspace(0, 1, precision)
    curve = np.zeros((len(ts), 2), dtype=np.float64)
    n = len(points) - 1

    for idx, t in enumerate(ts):
        for i in range(n + 1):
            bin_coef = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
            pin = bin_coef * (1 - t) ** (n - i) * t ** i
            curve[idx] += pin * points[i]

    return curve


def create_control_points(endpoints, max_vd, n_control_points):
    """Create the control points of a Bezier curve.

    Args:
        endpoints (list): The initial and final points of the curve
        max_vd (float): The maximum displacement of the control points. Sets the
        curvature of the curve.
        n_control_points (int): Number of control points

    Returns:
        control_points: List of control points
    """

    ps = endpoints[0]  # initial point
    pe = endpoints[1]  # final point
    dx = pe[0] - ps[0]
    dy = pe[1] - ps[1]
    distance = np.sqrt((pe[0] - ps[0]) ** 2 + (pe[1] - ps[1]) ** 2)
    normal_se = np.array((-dy, dx)) / distance  # or (dy, -dx) --> vector normal to (pe-ps)
    control_points = []
    hds = np.linspace(0.2, 0.8, n_control_points)

    for j in range(n_control_points):
        control_point = ((pe - ps) * hds[j])
        control_point += (normal_se * np.random.uniform(low=-1, high=1) * max_vd)
        control_points.append(control_point + ps)

    control_points.insert(0, ps)
    control_points.append(pe)

    return control_points


class VesselGeometry:

    def __init__(self, size, n_control_points, max_vd, radius, num_curves, extra_space=32):
        self.self = None
        self.size = size
        self.n_control_points = n_control_points
        self.max_vd = max_vd
        self.radius = radius
        self.num_curves = num_curves
        self.extra_space = extra_space

    def create_curve(self, img_shape, min_dist, max_dist, n_control_points, max_vd, precision, padding):
        """Create a Bezier curve.

        Args:
            img_shape (tuple): Number of rows and columns of the image
            min_dist (float): Minimum Euclidean distance between the endpoints of
            the curve
            max_dist (float): Maximum Euclidean distance between the endpoints of
            the curve
            n_control_points (int): Number of control points
            max_vd (float): The maximum displacement of the control points. Sets the
            curvature of the curve
            precision (int): Number of points to use to represent the curve
            padding (int): The endpoints of the curve will have a minimum distance
            equal to `padding` from the border of the image. This helps to avoid
            having parts of the curve outside the image

        Returns:
            curve: List of points representing the generated curve
            distance: The Euclidean distance between the endpoints of the curve
        """

        points, distance = create_endpoints(img_shape, min_dist, max_dist, padding)
        control_points = self.self.create_control_points(points, max_vd, n_control_points)
        curve = bezier(control_points, precision)

        return curve, distance

    def create_curves(self, size, n_control_points, max_vd, radius, num_curves, extra_space=32):
        """Create an artificial blood vessel image

        Args:
            size (int): Number of rows and columns of the image
            n_control_points (int): Number of control points for the Bezier curve
            max_vd (float): The maximum displacement of the control points. Sets the
            curvature of the curve
            radius (int): Radius of the curves
            num_curves (int): Number of curves to insert into the image
            extra_space (int): An image with size (size+2*extra_space, size+2*extra_space)
            is created and then cut to a size of (size, size). This simulates a real
            sample that captures part of a tissue.

        Returns:
            img: Image containing the curves
        """

        min_dist = size//2              # Minimum size of the curve
        max_dist = size+2*extra_space   # Maximum size of the curve

        img_shape = (size+2*extra_space, size+2*extra_space)

        img_curves = np.zeros(img_shape, dtype=np.uint8)

        for _ in range(num_curves):
            curve, _ = self.create_curve(img_shape, min_dist, max_dist, n_control_points, max_vd,
                                         precision=100, padding=0)
            curve = curve.astype(int)
            # Clip curve to avoid points outside the image
            curve[:,0] = np.clip(curve[:,0], 0, img_shape[0]-1)
            curve[:,1] = np.clip(curve[:,1], 0, img_shape[1]-1)

            # Draw curve
            img = np.zeros(img_shape, dtype=np.uint8)
            for idx in range(len(curve)-1):
                rr, cc = line(*curve[idx], *curve[idx+1])
                img[rr, cc] = 1

            img_dil = binary_dilation(img, iterations=radius)

            img_curves[img_dil>0] = 1

        img_curves = img_curves[extra_space:-extra_space, extra_space:-extra_space]

        return img_curves
