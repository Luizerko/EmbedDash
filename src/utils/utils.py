import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_distances


# Compute centroids for different clusters of data
def compute_centroids(data, labels):
    centroids = {}
    for i in np.unique(labels):
        mask = np.where(labels==i)
        centroids[i] = np.mean(data[mask], axis=0)

    return centroids

# Compute centroids pairwise distances
def compute_pairwise_distances(centroids):
    pairwise_matrix = pairwise_distances(centroids)
    pairwise_matrix = pairwise_matrix / np.max(pairwise_matrix)

    centroids_distances = {}
    for i in range(10):
        centroids_distances[i] = pairwise_matrix[i]

    return centroids_distances


# Objective function of optimization task (finding centroid positions that respect constraints imposed by latent space distances)
def objective_function(centroids, desired_distances):
    
    centroids = centroids.reshape(10, 2)
    current_distances = compute_pairwise_distances(centroids)
    
    error = []
    for k in desired_distances.keys():
        error.append(np.sum((current_distances[k] - desired_distances[k])**2))

    return np.sum(error)

# Compute translations of each cluster given the initial and final centroid positions
def compute_translations(initial_positions, final_positions):
    translations = {}
    for i, k in enumerate(initial_positions.keys()):
        translations[k] = final_positions[i] - initial_positions[k]

    return translations


# update CLICKED_POINTS to finde the points in between
def fit_line(coordinates):
    x_coords, y_coords = zip(*coordinates)

    if len(coordinates) == 2:
        poly = np.polyfit(x_coords, y_coords, 1)
        return 0, poly[0], poly[1]
    else:
        poly = np.polyfit(x_coords, y_coords, 2)
        return poly[0], poly[1], poly[2]


def shortest_distance(point, a, b, c):
    """
    Calculate the perpendicular distance from a point to a quadratic curve y = ax^2 + bx + c.

    Args:
    point (tuple): The (x, y) coordinates of the point.
    a (float): The coefficient a in the quadratic equation y = ax^2 + bx + c.
    b (float): The coefficient b in the quadratic equation y = ax^2 + bx + c.
    c (float): The coefficient c in the quadratic equation y = ax^2 + bx + c.

    Returns:
    float: The perpendicular distance from the point to the quadratic curve.
    """
    x0, y0 = point

    # Define the function for the distance squared (to avoid square root for simplicity)
    def distance_squared(x):
        y = a * x**2 + b * x + c
        return (x - x0) ** 2 + (y - y0) ** 2

    # Initial guess for the x-coordinate of the point on the curve closest to (x0, y0)
    initial_guess = x0

    # Minimize the distance squared to find the closest point on the curve
    result = minimize(distance_squared, initial_guess)
    x_closest = result.x[0]
    y_closest = a * x_closest**2 + b * x_closest + c

    # Calculate the Euclidean distance
    distance = np.sqrt((x_closest - x0) ** 2 + (y_closest - y0) ** 2)
    return distance

def closest_points_to_line(coordinates, images, labels, threshold):
    a, b, c = fit_line(coordinates) # from ax**2 + bx + c

    closest_points = []
    distances = []
    for i, point in enumerate(coordinates):
        distance = shortest_distance(point, a, b, c)
        distances.append(distance)
        # if distance > threshold:
        #     raise ValueError(f'{distance, threshold}')
        if distance <= threshold:
            # raise ValueError(f'{distance, threshold}')
            closest_points.append(((images[i], labels[i]), point))
    # raise ValueError(f'{closest_points}, coords: {coordinates}, distances: {distances}')
    return closest_points

def sort_coordinates_with_others(coords, *other_lists):
    """
    Sort a list of (x, y) coordinates and other lists based on the sorted coordinates.
    
    Args:
    coords (list of tuples): List of (x, y) coordinates.
    other_lists (list of lists): Additional lists to be sorted in the same order as coords.
    
    Returns:
    tuple: Sorted list of coordinates and other lists.
    """
    # Combine the coordinates with their corresponding other list elements
    combined = list(zip(coords, *other_lists))

    # Sort the combined list by the coordinates
    combined_sorted = sorted(combined, key=lambda item: (item[0][0], item[0][1]))

    # Unzip the sorted combined list
    sorted_coords, *sorted_other_lists = zip(*combined_sorted)

    # Convert the sorted_coords and sorted_other_lists to lists
    sorted_coords = list(sorted_coords)
    sorted_other_lists = [list(lst) for lst in sorted_other_lists]

    return sorted_coords, sorted_other_lists
