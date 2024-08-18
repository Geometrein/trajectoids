import os
import logging
from math import atan2
from typing import Union, Tuple, List

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import brentq
from scipy import interpolate
from skimage import io
import trimesh
import open3d as o3d
import plotly.graph_objects as go


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def numpy_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the cross product of two 3D vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        np.ndarray: The cross product of the two vectors.
    """
    return np.cross(a, b)


def numpy_dot_sign(a: np.ndarray, b: np.ndarray) -> int:
    """
    Computes the sign of the dot product of two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        int: 1 if the dot product is positive, -1 if negative, 0 if zero.
    """
    dot_product = np.dot(a, b)
    return (dot_product > 0) - (dot_product < 0)


def numpy_intersects(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    """
    Determines if two line segments intersect in 3D space.

    Args:
        a (np.ndarray): First point of the first segment.
        b (np.ndarray): Second point of the first segment.
        c (np.ndarray): First point of the second segment.
        d (np.ndarray): Second point of the second segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    ab_cross_product = numpy_cross(a, b)
    cdx_cross_product = numpy_cross(c, d)
    combined_cross_product = numpy_cross(ab_cross_product, cdx_cross_product)

    signs = (
        numpy_dot_sign(numpy_cross(ab_cross_product, a), combined_cross_product)
        + numpy_dot_sign(numpy_cross(b, ab_cross_product), combined_cross_product)
        + numpy_dot_sign(numpy_cross(cdx_cross_product, c), combined_cross_product)
        + numpy_dot_sign(numpy_cross(d, cdx_cross_product), combined_cross_product)
    )

    return abs(signs) == 4


def invalid_value_check(path):
    if np.isnan(path).any() or np.isinf(path).any():
        raise ValueError("Input path contains NaN or Inf values.")


def sort_by_first_column(
    matrix: NDArray[Union[int, float]],
) -> NDArray[Union[int, float]]:
    """
    Sort a 2D array based on the values in the first column.

    Args:
        matrix (NDArray[Union[int, float]]): A 2D numpy array to be sorted.

    Returns:
        NDArray[Union[int, float]]: The sorted 2D numpy array.
    """
    return matrix[np.argsort(matrix[:, 0])]


def calculate_signed_angle_2d(vec1: NDArray[float], vec2: NDArray[float]) -> float:
    """
    Calculate the signed angle between two 2-dimensional vectors using the atan2 formula.

    The angle is positive if rotation from vec1 to vec2 is counterclockwise, and negative
    if the rotation is clockwise. The angle is in radians.

    Args:
        vec1 (NDArray[float]): The first 2D vector.
        vec2 (NDArray[float]): The second 2D vector.

    Returns:
        float: The signed angle between the two vectors in radians.
    """
    assert vec1.shape == (2,), "vec1 must be a 2-dimensional vector."
    assert vec2.shape == (2,), "vec2 must be a 2-dimensional vector."

    # Normalize the 2D vectors and convert them to 3D for cross product computation
    vec1_3d = np.append(vec1, 0) / np.linalg.norm(vec1)
    vec2_3d = np.append(vec2, 0) / np.linalg.norm(vec2)
    return atan2(np.cross(vec1_3d, vec2_3d)[-1], np.dot(vec1_3d, vec2_3d))


def unsigned_angle_between_vectors(vector1, vector2):
    """Calculate the unsigned angle between two n-dimensional vectors using the atan2 formula.
    Angle is in radians.

    This is more numerically stable for angles close to 0 or pi than the acos() formula.
    """
    return atan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))


def rotate_2d(vector, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = (0, 0)
    px, py = vector

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])


def rotate_3d_vector(
    vector: NDArray[float], axis: NDArray[float], angle: float
) -> NDArray[float]:
    """
    Rotate a 3D vector around a specified axis by a given angle.

    Args:
        vector (NDArray[float]): The 3D vector to rotate.
        axis (NDArray[float]): The axis of rotation (a 3D vector).
        angle (float): The angle of rotation in radians.

    Returns:
        NDArray[float]: The rotated 3D vector.
    """
    point_cloud = trimesh.PointCloud([vector])
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=angle, direction=axis, point=[0, 0, 0]
    )
    point_cloud.apply_transform(rotation_matrix)
    return np.array(point_cloud.vertices[0])


def spherical_trace_is_self_intersecting(sphere_trace: NDArray[float]) -> bool:
    """
    Check if a spherical trace is self-intersecting.

    Args:
        sphere_trace (NDArray[float]): A 2D numpy array representing a sequence of points on a sphere.

    Returns:
        bool: True if the trace is self-intersecting, False otherwise.
    """
    arcs = [
        [sphere_trace[i], sphere_trace[i + 1]] for i in range(sphere_trace.shape[0] - 1)
    ]

    for i in range(len(arcs)):
        for j in range(len(arcs) - 1, i + 1, -1):
            # Avoid comparing adjacent arcs and the first and last arc in the loop
            if not ((j <= i + 1) or (i == 0 and j == len(arcs) - 1)):
                if numpy_intersects(arcs[i][0], arcs[i][1], arcs[j][0], arcs[j][1]):
                    LOGGER.info(f"Self-intersection detected between arc {i} and arc {j}")
                    return True
    return False


def get_trajectory_from_raster_image(
    filename: str, resample_to: int = 200
) -> NDArray[float]:
    """
    Extracts a trajectory from a raster image, resamples it, and optionally plots it.

    Args:
        filename (str): The path to the raster image file.
        resample_to (int): The number of points to resample the trajectory to. Defaults to 200.

    Returns:
        NDArray[float]: A 2D numpy array representing the trajectory points.
    """
    # Load the image and preprocess
    image = io.imread(filename)[:, :, 0].T
    image = np.fliplr(image)

    # Initialize lists for x and y coordinates
    x_coords = []
    y_coords = []

    # Extract trajectory from the image
    for i in range(image.shape[0]):
        if np.all(image[i, :] == 255):  # Skip rows where all pixels are white
            continue
        x_coords.append(i / image.shape[0] * 2 * np.pi)  # Assume x dimension is 2*pi
        y_coords.append(
            np.mean(np.argwhere(image[i, :] != 255)) / image.shape[0] * 2 * np.pi
        )

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Normalize the x coordinates
    x_coords -= x_coords[0]

    # Resample the trajectory if specified
    if resample_to is not None:
        x_resampled = np.linspace(x_coords[0], x_coords[-1], resample_to)
        y_coords = interpolate.interp1d(x_coords, y_coords, fill_value="extrapolate")(
            x_resampled
        )
        x_coords = x_resampled

    # Normalize the y coordinates
    y_coords -= y_coords[0]
    y_coords -= x_coords * (y_coords[-1] - y_coords[0]) / (x_coords[-1] - x_coords[0])

    # Stack the x and y coordinates into a 2D array
    trajectory_points = np.stack((x_coords, y_coords)).T

    # Plot the trajectory
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
    plt.axis("equal")
    plt.show()

    return trajectory_points


def get_trajectory_from_csv(filename: str) -> NDArray[float]:
    """
    Loads trajectory data from a CSV file.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        NDArray[float]: A 2D numpy array where each row represents a point in the trajectory.
    """
    try:
        data = np.genfromtxt(filename, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, len(data))  # Ensure the data is at least 2D
        return data
    except Exception as e:
        raise ValueError(f"Error reading the CSV file '{filename}': {e}")


def rotation_from_point_to_point(
    point: NDArray[float], previous_point: NDArray[float]
) -> Tuple[NDArray[float], float]:
    """
    Calculate the rotation matrix and angle required to rotate from one point to another in 3D space.

    Args:
        point (NDArray[float]): The current point in 3D space.
        previous_point (NDArray[float]): The previous point in 3D space.

    Returns:
        Tuple[NDArray[float], float]: A tuple containing the rotation matrix (4x4) and the rotation angle in radians.
    """
    # Calculate the vector pointing from the current point to the previous point
    vector_to_previous_point = previous_point - point

    # Define the axis of rotation as perpendicular to the vector
    axis_of_rotation = np.array(
        [vector_to_previous_point[1], -vector_to_previous_point[0], 0]
    )

    # Calculate the rotation angle as the magnitude of the vector
    theta = np.linalg.norm(vector_to_previous_point)

    # Create the rotation matrix using the axis of rotation and the angle
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=-theta, direction=axis_of_rotation, point=[0, 0, 0]
    )

    return rotation_matrix, theta


def rotation_to_previous_point(
    i: int, data: NDArray[float]
) -> Tuple[NDArray[float], float]:
    """
    Calculate the rotation matrix and angle required to rotate from the current point
    to the previous point in a 3D trajectory.

    Args:
        i (int): The index of the current point in the trajectory data.
        data (NDArray[float]): A 2D numpy array representing the trajectory points (Nx3).

    Returns:
        Tuple[NDArray[float], float]: A tuple containing the rotation matrix (4x4) and the rotation angle in radians.
    """
    # Current point
    point = data[i]

    # Previous point
    previous_point = data[i - 1]

    # Calculate the rotation matrix and angle to the previous point
    return rotation_from_point_to_point(point, previous_point)


def rotation_to_origin(
    index_in_trajectory: int, data: NDArray[float]
) -> NDArray[float]:
    """
    Compute the rotation matrix that aligns the current point in the trajectory to the origin.

    Args:
        index_in_trajectory (int): The index of the current point in the trajectory.
        data (NDArray[float]): A 2D numpy array representing the trajectory points (Nx3).

    Returns:
        NDArray[float]: A 4x4 numpy array representing the rotation matrix to align the point to the origin.
    """
    if index_in_trajectory == 0:
        net_rotation_matrix = trimesh.transformations.identity_matrix()
    else:
        net_rotation_matrix, theta = rotation_to_previous_point(
            index_in_trajectory, data
        )
        net_rotation_matrix = trimesh.transformations.concatenate_matrices(
            rotation_to_origin(index_in_trajectory - 1, data), net_rotation_matrix
        )
    return net_rotation_matrix


def compute_shape(
    input_data: NDArray[float],
    kx: float,
    ky: float,
    output_folder: str = "outputs",
    core_radius: float = 1,
    cut_size: float = 10,
) -> None:
    """
    Compute the shape by transforming a trajectory and generating cutting boxes, which are then saved as STL files.

    Args:
        data0 (NDArray[float]): The input trajectory data as a 2D numpy array.
        kx (float): Scaling factor for the x-axis.
        ky (float): Scaling factor for the y-axis.
        output_folder (str): The folder path where the trajectory data will be saved.
        core_radius (float): The radius of the core used for generating the cutting boxes. Defaults to 1.
        cut_size (float): The size of the cutting boxes relative to the core radius. Defaults to 10.

    Returns:
        None
    """
    # Scale the trajectory data
    data = np.copy(input_data)
    data[:, 0] *= kx
    np.savetxt(f"{output_folder}/data.csv", data, delimiter=",")
    data[:, 1] *= ky

    # Compute the rotation of the entire trajectory
    LOGGER.info(f"Number of points in the trajectory: {data.shape[0] - 1}")
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(
        rotation_to_origin(data.shape[0] - 1, data)
    )
    LOGGER.info(f"Rotation of entire trajectory: {rotation_of_entire_traj}")

    # Save the transformed path data
    np.save(f"{output_folder}/path_data.npy", data)

    # Create the base box for cutting
    base_box = trimesh.creation.box(
        extents=[
            cut_size * core_radius,
            cut_size * core_radius,
            cut_size * core_radius,
        ],
        transform=trimesh.transformations.translation_matrix(
            [0, 0, -core_radius - 1 * cut_size * core_radius / 2]
        ),
    )

    # Generate cutting boxes based on the trajectory
    boxes_for_cutting: List[trimesh.Trimesh] = []
    for i, point in enumerate(data):
        box_for_cutting = base_box.copy()
        box_for_cutting.apply_transform(rotation_to_origin(i, data))
        boxes_for_cutting.append(box_for_cutting)

    if not os.path.exists(f"{output_folder}/cut_meshes/"):
        os.makedirs(f"{output_folder}/cut_meshes/")
    # Export each cutting box as an STL file
    for i, box in enumerate(boxes_for_cutting):
        box.export(f"{output_folder}/cut_meshes/mesh_{i}.stl")


def trace_on_sphere(
    input_data: NDArray[float],
    kx: float,
    ky: float,
    core_radius: float = 1,
    do_plot: bool = False,
) -> NDArray[float]:
    """
    Compute the trace of a trajectory on the surface of a sphere by applying transformations based on the trajectory data.

    Args:
        data0 (NDArray[float]): The input trajectory data as a 2D numpy array.
        kx (float): Scaling factor for the x-axis.
        ky (float): Scaling factor for the y-axis.
        core_radius (float): The radius of the sphere. Defaults to 1.
        do_plot (bool): Whether to plot the trace. Defaults to False.

    Returns:
        NDArray[float]: A 2D numpy array representing the trace points on the sphere.
    """
    # Scale the trajectory data
    data = np.copy(input_data)
    data[:, 0] *= kx
    data[:, 1] *= ky

    # Initialize the point at the plane and the list to store the trace on the sphere
    point_at_plane = trimesh.PointCloud([[0, 0, -core_radius]])
    sphere_trace = []

    # Apply transformations to trace the trajectory on the sphere
    for i, point in enumerate(data):
        point_at_plane_copy = point_at_plane.copy()
        point_at_plane_copy.apply_transform(rotation_to_origin(i, data))
        sphere_trace.append(np.array(point_at_plane_copy.vertices[0]))

    # Convert the list to a numpy array
    sphere_trace = np.array(sphere_trace)

    # Optionally plot the trace
    if do_plot:
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot3D(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    return sphere_trace


def mismatch_angle_for_path(input_path: NDArray[float]) -> float:
    """
    Calculate the mismatch angle for the given path by determining the rotation
    of the entire trajectory from the origin.

    Args:
        input_path (NDArray[float]): A 2D numpy array representing the path.

    Returns:
        float: The mismatch angle in radians.
    """
    rotation_matrix = trimesh.transformations.rotation_from_matrix(
        rotation_to_origin(input_path.shape[0] - 1, input_path)
    )
    angle = rotation_matrix[0]
    return angle


def minimize_mismatch_by_scaling(input_path_0, scale_range=(0.8, 1.2)):
    scale_max = scale_range[1]
    scale_min = scale_range[0]
    # if the sign of mismatch angle is same at the ends of the region -- there is no solution
    if mismatch_angle_for_path(input_path_0 * scale_max) * mismatch_angle_for_path(input_path_0 * scale_min) > 0:
        LOGGER.info('Sign of mismatch is the same on both sides of the interval.')
        LOGGER.info(f'Mismatch at max scale = {mismatch_angle_for_path(input_path_0 * scale_min)}')
        LOGGER.info(f'Mismatch at min scale = {mismatch_angle_for_path(input_path_0 * scale_max)}')
        return False

    def left_hand_side(x):  # the function whose root we want to find
        LOGGER.debug(f'Sampling function at x={x}')
        return mismatch_angle_for_path(input_path_0 * x)

    best_scale = brentq(left_hand_side, a=scale_min, b=scale_max, maxiter=80, xtol=0.00001, rtol=0.00005)
    LOGGER.debug(f'Minimized mismatch angle = {left_hand_side(best_scale)}')
    return best_scale


def double_the_path(input_path_0: NDArray[float]) -> NDArray[float]:
    """
    Double the path by appending a second path with adjusted x-coordinates to the original path.

    Args:
        input_path_0 (NDArray[float]): A 2D numpy array representing the initial path.

    Returns:
        NDArray[float]: The doubled path as a 2D numpy array.
    """
    # Create the second path by adjusting the x-coordinates
    input_path_1 = np.copy(input_path_0)
    input_path_1[:, 0] += input_path_1[-1, 0]
    # Concatenate the paths and sort by the first column
    input_path_combined = np.concatenate(
        (input_path_0, sort_by_first_column(input_path_1)[1:,]), axis=0
    )
    input_path_combined = sort_by_first_column(input_path_combined)
    return input_path_combined


def plot_input_path(input_path_single_section: NDArray[float]) -> None:
    """
    Plots a 2D path after adjusting it by normalizing and correcting its slope.

    Args:
        input_path_single_section (NDArray[float]): A 2D numpy array representing the path.
                                                    Each row corresponds to a point (x, y).

    Returns:
        None
    """
    # Extract X and Y coordinates
    xs = input_path_single_section[:, 0]
    ys = input_path_single_section[:, 1]

    # Normalize X and Y coordinates to start from zero
    xs = xs - xs[0]
    ys = ys - ys[0]

    # Correct the Y coordinates by removing the linear trend
    ys = ys - xs * (ys[-1] - ys[0]) / (xs[-1] - xs[0])

    # Update the path with normalized and corrected coordinates
    input_path_single_section[:, 0] = xs
    input_path_single_section[:, 1] = ys

    # Plot the adjusted path
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(input_path_single_section[:, 0], input_path_single_section[:, 1])
    plt.axis("equal")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.title("Path loaded from the CSV file")
    plt.grid()
    plt.show()


def cumsum_half_length_along_the_path(input_path_0: NDArray[float]) -> NDArray[float]:
    """
    Calculate the cumulative sum of half the length along the path, normalized to [0, 1].

    Args:
        input_path_0 (NDArray[float]): A 2D numpy array representing the path.

    Returns:
        NDArray[float]: A 1D numpy array representing the cumulative length along the path.
    """
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]

    # Calculate the cumulative length along the path
    length_along_path = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    length_along_path = np.insert(length_along_path, 0, 0)
    length_along_path = np.remainder(length_along_path, np.max(length_along_path) / 2)
    length_along_path /= np.max(length_along_path)

    return length_along_path


def upsample_path(
    input_path: NDArray[float], by_factor: int = 10, kind: str = "linear"
) -> NDArray[float]:
    """
    Upsample the input path by a specified factor using interpolation.

    Args:
        input_path (NDArray[float]): A 2D numpy array representing the original path.
        by_factor (int): The factor by which to upsample the path. Defaults to 10.
        kind (str): The kind of interpolation to use ('linear', 'cubic', etc.). Defaults to 'linear'.

    Returns:
        NDArray[float]: A 2D numpy array representing the upsampled path.
    """
    old_indices = np.arange(input_path.shape[0])
    max_index = input_path.shape[0] - 1
    new_indices = np.arange(0, max_index, 1 / by_factor)
    new_indices = np.append(new_indices, max_index)

    new_xs = interpolate.interp1d(old_indices, input_path[:, 0], kind=kind)(new_indices)
    new_ys = interpolate.interp1d(old_indices, input_path[:, 1], kind=kind)(new_indices)

    return np.stack((new_xs, new_ys)).T


def get_gauss_bonnet_area(input_path, flat_path_change_of_direction='auto', return_arc_normal=False):
    '''This function does not take into account the possibly changing rotation index of the spherical trace.
    It has to be accounted for in the downstream code.'''
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)

    # Change of direction of the flat path:
    if flat_path_change_of_direction == 'auto':
        flat_path_change_of_direction = np.sum(
            np.array([calculate_signed_angle_2d(input_path[i + 2] - input_path[i + 1],
                                                      input_path[i + 1] - input_path[i])
                      for i in range(input_path.shape[0] - 2)
                      ]))

    # Change of direction due to 2 angles formed by great arc connecting the last and first point
    path_start_direction_vector_flat = input_path[1] - input_path[0]
    path_start_direction_vector_flat /= np.linalg.norm(path_start_direction_vector_flat)
    path_start_direction_vector = np.array([path_start_direction_vector_flat[0],
                                            path_start_direction_vector_flat[1],
                                            0])
    path_start_arc_normal = np.cross(np.array([0, 0, -1]), path_start_direction_vector)
    path_start_arc_normal /= np.linalg.norm(path_start_arc_normal)

    # direction from point (-2) to point (-1), expressed as normal of the respective arc on the sphere_trace
    path_end_direction_vector_flat = input_path[-1] - input_path[-2]
    path_end_direction_vector_flat /= np.linalg.norm(path_end_direction_vector_flat)

    # Convert to trimesh point cloud and apply reverse rolling to origin
    point_at_plane = trimesh.PointCloud([[path_end_direction_vector_flat[0],
                                          path_end_direction_vector_flat[1],
                                          -1]])
    point_at_plane.apply_transform(rotation_to_origin(input_path.shape[0] - 1, input_path))
    path_end_direction_vector = point_at_plane.vertices[0] - sphere_trace[-1, :]

    # compute the normal of that rotation arc
    path_end_arc_normal = np.cross(sphere_trace[-1, :], path_end_direction_vector)
    path_end_arc_normal /= np.linalg.norm(path_end_arc_normal)

    # Now calculate the change of direction due to the arc connecting the trace ends
    normal_of_arc_connecting_trace_ends = np.cross(sphere_trace[-1], sphere_trace[0])
    normal_of_arc_connecting_trace_ends /= np.linalg.norm(normal_of_arc_connecting_trace_ends)

    def get_signed_change_of_direction_at_point(first_arc_axis, second_arc_axis, central_point):
        unsigned_sine = np.cross(first_arc_axis, second_arc_axis)
        signed_sine = np.linalg.norm(unsigned_sine) * np.sign(np.dot(unsigned_sine,
                                                                     central_point))
        signed_cosine = np.dot(first_arc_axis, second_arc_axis)
        return np.arctan2(signed_sine, signed_cosine)

    # change of direction computed from the flat path angles
    net_change_of_direction = flat_path_change_of_direction
    logging.debug(f'Flat path change of direction = {net_change_of_direction / np.pi} pi')

    # change of direction from end of trace to the connecting arc
    net_change_of_direction += get_signed_change_of_direction_at_point(path_end_arc_normal,
                                                                       normal_of_arc_connecting_trace_ends,
                                                                       sphere_trace[-1])
    logging.debug(f'Net change of direction path with first angle to arc = {net_change_of_direction / np.pi} pi')

    # change of direction from connecting arc to the start of trace
    net_change_of_direction += get_signed_change_of_direction_at_point(normal_of_arc_connecting_trace_ends,
                                                                       path_start_arc_normal,
                                                                       sphere_trace[0])

    gauss_bonnet_area = 2 * np.pi - net_change_of_direction
    logging.debug(f'Net change of direction = {net_change_of_direction / np.pi} pi')
    logging.debug(f'Area = {gauss_bonnet_area / np.pi} pi')

    if return_arc_normal:
        end_to_end_distance = np.linalg.norm(sphere_trace[0] - sphere_trace[-1])
        return gauss_bonnet_area, normal_of_arc_connecting_trace_ends, end_to_end_distance
    else:
        return gauss_bonnet_area


def gauss_bonnet_areas_for_all_scales(
    input_path: np.ndarray,
    minscale: float = 0.01,
    maxscale: float = 2,
    nframes: int = 100,
    exclude_legitimate_discont: bool = False,
    adaptive_sampling: bool = True,
    diff_thresh: float = 2 * np.pi * 0.1,
    max_number_of_subdivisions: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Bonnet areas for various scales of a given path on a sphere.

    This function evaluates the Gauss-Bonnet theorem over different scales of an input spherical path.
    The function computes the area enclosed by the path, considering possible changes in the
    rotation index of the spherical trace. Optionally, it can adaptively sample scales based on
    the difference threshold to improve accuracy.

    Parameters:
    ----------
    input_path : np.ndarray
        The 2D or 3D input path represented as a numpy array.
    minscale : float, optional
        The minimum scale factor to be applied to the input path (default is 0.01).
    maxscale : float, optional
        The maximum scale factor to be applied to the input path (default is 2).
    nframes : int, optional
        The number of scale frames to evaluate between minscale and maxscale (default is 100).
    exclude_legitimate_discont : bool, optional
        Whether to exclude legitimate discontinuities when calculating area (default is False).
    adaptive_sampling : bool, optional
        If True, scales will be adaptively sampled based on the difference threshold (default is True).
    diff_thresh : float, optional
        The threshold for the difference in areas to trigger additional sampling (default is 2 * np.pi * 0.1).
    max_number_of_subdivisions : int, optional
        The maximum number of subdivisions allowed during adaptive sampling (default is 15).

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the array of sweeped scales and the corresponding Gauss-Bonnet areas.

    """
    gauss_bonnet_areas = []
    connecting_arc_axes = []
    end_to_end_distances = []
    sweeped_scales = np.linspace(minscale, maxscale, nframes)

    flat_path_change_of_direction = np.sum(
        np.array([calculate_signed_angle_2d(input_path[i + 2] - input_path[i + 1],
                                            input_path[i + 1] - input_path[i])
                  for i in range(input_path.shape[0] - 2)
                  ]))
    for frame_id, scale in enumerate(sweeped_scales):
        LOGGER.debug(f'Computing GB_area for scale {scale}')
        input_path_scaled = input_path * scale
        gb_area_here, arc_axis, end_to_end = get_gauss_bonnet_area(input_path_scaled,
                                                         flat_path_change_of_direction,
                                                         return_arc_normal=True)
        gauss_bonnet_areas.append(gb_area_here)
        connecting_arc_axes.append(arc_axis)
        end_to_end_distances.append(end_to_end)

    end_to_end_distances = np.array(end_to_end_distances)

    if adaptive_sampling:
        for subdivision_iteration in range(max_number_of_subdivisions):
            LOGGER.debug(f'Subdivision iteration: {subdivision_iteration}')
            area_diff = np.diff(gauss_bonnet_areas)
            if np.max(area_diff) < diff_thresh:
                break
            insert_before_indices = []
            insert_scales = []
            insert_areas = []
            insert_axes = []
            insert_ends = []
            for i, area in enumerate(gauss_bonnet_areas[:-1]):
                if np.abs(area_diff[i]) > diff_thresh:
                    insert_before_indices.append(i + 1)
                    new_scale_here = (sweeped_scales[i] + sweeped_scales[i + 1]) / 2
                    LOGGER.debug(f'Sampling at new scale {new_scale_here}')
                    insert_scales.append(new_scale_here)
                    if not exclude_legitimate_discont:
                        gb_area_here, arc_axis, end_to_end = get_gauss_bonnet_area(input_path * new_scale_here,
                                                                         flat_path_change_of_direction,
                                                                         return_arc_normal=True)
                    else:
                        gb_area_here = get_gauss_bonnet_area(input_path * new_scale_here,
                                                   flat_path_change_of_direction,
                                                   return_arc_normal=False)
                    insert_areas.append(gb_area_here)
                    insert_axes.append(arc_axis)
                    insert_ends.append(end_to_end)

            sweeped_scales = np.insert(sweeped_scales, insert_before_indices, insert_scales)
            gauss_bonnet_areas = np.insert(gauss_bonnet_areas, insert_before_indices, insert_areas)
            if not exclude_legitimate_discont:
                end_to_end_distances = np.insert(end_to_end_distances, insert_before_indices, insert_ends)
                acc = 0
                for i in range(len(insert_axes)):
                    connecting_arc_axes.insert(insert_before_indices[i] + acc, insert_axes[i])
                    acc += 1

    gb_areas = np.array(gauss_bonnet_areas)
    connecting_arc_axes = tuple(connecting_arc_axes)
    gb_area_zero = round(gb_areas[0] / np.pi) * np.pi
    gb_areas -= gb_area_zero
    LOGGER.info(f'Initial Gauss-Bonnet area is {gb_area_zero / np.pi} pi')

    additional_rotation_indices = np.zeros_like(gb_areas)
    additional_rotation_index_here = 0
    threshold_for_ind = 2 * np.pi * 0.75
    for i in range(1, gb_areas.shape[0]):
        diff_here = gb_areas[i] - gb_areas[i - 1]
        if np.abs(diff_here) > threshold_for_ind:
            if exclude_legitimate_discont and \
                    ((np.dot(connecting_arc_axes[i], connecting_arc_axes[i - 1]) < 0) and (
                            end_to_end_distances[i] > 1.4) and (end_to_end_distances[i - 1] > 1.4)):
                LOGGER.info(f'Legitimate discontinuity of area is found at scale {sweeped_scales[i]}')
            else:
                additional_rotation_index_here += np.round(diff_here / (2 * np.pi))
        additional_rotation_indices[i] = additional_rotation_index_here
    gb_areas -= 2 * np.pi * additional_rotation_indices
    return sweeped_scales, gb_areas


def plot_trajectoid(stl_path: str = "trajectoid.stl") -> None:
    """
    Plots a 3D mesh from an STL file using Plotly.

    Args:
        stl_path (str): The path to the STL file. Defaults to "trajectoid.stl".

    Returns:
        None
    """
    # Load the mesh from the STL file
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty. Check the file at {stl_path}.")

    # Ensure the mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Extract triangles and vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Set colors based on triangle normals if available
    if mesh.has_triangle_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
        colors = tuple(map(tuple, colors))  # Convert to a tuple of tuples for Plotly
    else:
        colors = (1.0, 0.0, 0.0)

    # Create a 3D plot using Plotly
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                facecolor=colors,
                opacity=1,
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        ),
    )
    fig.update_layout(width=1600, height=900, autosize=False)
    fig.show()


def plot_animated_trajectoid(stl_path: str = "trajectoid.stl") -> None:
    """
    Plots a 3D mesh from an STL file using Plotly with automatic rotation animation.

    Args:
        stl_path (str): The path to the STL file. Defaults to "trajectoid.stl".

    Returns:
        None
    """
    # Load the mesh from the STL file
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty. Check the file at {stl_path}.")

    # Ensure the mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Extract triangles and vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Set colors based on triangle normals if available
    if mesh.has_triangle_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
        colors = tuple(map(tuple, colors))  # Convert to a tuple of tuples for Plotly
    else:
        colors = (1.0, 0.0, 0.0)

    # Create the 3D plot using Plotly
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                facecolor=colors,
                opacity=1,
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=2, y=2, z=1.5)
                )
            ),
            width=1600,
            height=900,
            autosize=False,
        )
    )

    # Generate animation frames
    frames = []
    for angle in np.linspace(0, 360, 120):
        camera = dict(
            eye=dict(
                x=2 * np.cos(np.radians(angle)),
                y=2 * np.sin(np.radians(angle)),
                z=1.5
            )
        )
        frames.append(go.Frame(layout=dict(scene_camera=camera)))

    fig.frames = frames

    # Animation settings
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=50, redraw=True),
                                           fromcurrent=True,
                                           mode='immediate')])]
        )]
    )

    # Show the plot
    fig.show()


def plot_flat_path_with_color(
    input_path: NDArray[float],
    linewidth: int = 1,
    alpha: float = 1.0,
) -> None:
    """
    Plot a flat path with color based on the cumulative length along the path.

    Args:
        input_path (NDArray[float]): A 2D numpy array representing the path.
        half_of_input_path (NDArray[float]): A 2D numpy array representing half of the path.
        axs (plt.Axes): The matplotlib axes on which to plot.
        linewidth (int): The width of the line. Defaults to 1.
        alpha (float): The alpha (transparency) level of the line. Defaults to 1.
        plot_single_period (bool): Whether to plot only a single period of the path. Defaults to False.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 1, figsize=(16, 9))
    length_from_start_to_here = cumsum_half_length_along_the_path(input_path)
    half_of_input_path = input_path[: int(input_path.shape[0] / 2), :]

    x = input_path[:, 0]
    y = input_path[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Coloring the curve
    norm = plt.Normalize(
        length_from_start_to_here.min(), length_from_start_to_here.max()
    )
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(length_from_start_to_here)
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    axs.add_collection(lc)

    # black dots at middle and ends of the path
    for point in [half_of_input_path[0], half_of_input_path[-1], input_path[-1]]:
        axs.scatter(point[0], point[1], color="black", s=10)
    plt.axis("equal")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.title("Doubled Path with Color")
    plt.grid()
    plt.show()


def plot_spherical_trace_with_color_along_the_trace(
    input_path: NDArray[float], scale: float, plotting_upsample_factor: int = 1
) -> None:
    """
    Plot the spherical trace of a path with color along the trace based on the cumulative length.

    Args:
        input_path (NDArray[float]): A 2D numpy array representing the path.
        scale (float): The scaling factor for the path.
        plotting_up sample_factor (int): The factor by which to up sample the path for smoother plotting. Defaults to 1.

    Returns:
        None
    """
    length_from_start_to_here = cumsum_half_length_along_the_path(input_path)
    sphere_trace = trace_on_sphere(
        upsample_path(scale * input_path, by_factor=plotting_upsample_factor),
        kx=1,
        ky=1,
    )

    LOGGER.debug("Mlab plot begins...")

    fig = go.Figure(
        data=go.Scatter3d(
            x=sphere_trace[:, 0],
            y=sphere_trace[:, 1],
            z=sphere_trace[:, 2],
            marker=dict(
                size=0,
                color=length_from_start_to_here,
                colorscale="Viridis",
            ),
            line=dict(color=length_from_start_to_here, colorscale="Viridis", width=5),
        )
    )

    fig.update_layout(
        width=800,
        height=700,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                ),
            ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="manual",
        ),
    )

    fig.show()
