import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_distances

import plotly.express as px
import plotly.graph_objects as go
from layouts import fig_layout_dict, small_fig_layout_dict, fig_layout_dict_mammoth, fig_layout_dict_latent


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


def make_mnist_figure(df, version, index=False, is_subplot=False):

    main_type = version.split('_')[0]
    category_order = sorted(df['label'].unique())

    if main_type == "trimap":
        updated_fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="TRIMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=800, height=640, size_max=10,
            category_orders={'label': category_order}
        )
        
    elif main_type == "umap":
        updated_fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="UMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=800, height=640, size_max=10,
            category_orders={'label': category_order}
        )
        
    elif main_type == "tsne":
        updated_fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="T-SNE ",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=800, height=640, size_max=10,
            category_orders={'label': category_order}
        )

    elif main_type == "pacmap":
        updated_fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="PACMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=800, height=640, size_max=10,
            category_orders={'label': category_order}
        )
    
    # if no point is highlighted, return the figure

    if not index:

        if is_subplot:
            updated_fig.update_layout(small_fig_layout_dict)
        else:
            updated_fig.update_layout(fig_layout_dict)
            
        return updated_fig

    # if a point is highlighted, highlight that point
    df_row = df[df['index'] == index]
    x = float(df_row['x_'+version])
    y = float(df_row['y_'+version])
    
    marker = go.Scattergl(
        x=[x], y=[y],
        mode='markers',
        marker=dict(
            size=20,
            color='rgba(0, 0, 0, 1)',
            symbol='diamond-open',
            line=dict(color='rgba(0,0,0,1)', width=5)
        ),
        name='hover_marker',
        showlegend=False
    )

    updated_fig.add_trace(marker)

    if is_subplot:
        updated_fig.update_layout(small_fig_layout_dict)
    else:
        updated_fig.update_layout(fig_layout_dict)
    
    return updated_fig


def make_mammoth_figure(df, version, index=False):

    main_type = version.split('_')[0]

    if main_type == "original":
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='label',
            title="Original Mammoth Data",
            hover_data={'x': False, 'y': False, 'z': False, 'index':False, 'label': False},
            width=600, height=480
        )

    elif main_type == "trimap":
        fig = px.scatter_3d(
            df, x='x_'+version, y='y_'+version, z='z_'+version, color='label',
            title="TRIMAP",
            hover_data={'x_'+version: False, 'y_'+version: False, 'z_'+version: False, 'index':False, 'label': False},
            width=600, height=480
        )
    
    elif main_type == "umap":
        fig = px.scatter_3d(
            df, x='x_'+version, y='y_'+version, z='z_'+version, color='label',
            title="UMAP",
            hover_data={'x_'+version: False, 'y_'+version: False, 'z_'+version: False, 'index':False, 'label': False},
            width=600, height=480
        )
        
    elif main_type == "tsne":
        fig = px.scatter_3d(
            df, x='x_'+version, y='y_'+version, z='z_'+version, color='label',
            title="T-SNE",
            hover_data={'x_'+version: False, 'y_'+version: False, 'z_'+version: False, 'index':False, 'label': False},
            width=600, height=480
        )

    elif main_type == "pacmap":
        fig = px.scatter_3d(
            df, x='x_'+version, y='y_'+version, z='z_'+version, color='label',
            title="PACMAP",
            hover_data={'x_'+version: False, 'y_'+version: False, 'z_'+version: False, 'index':False, 'label': False},
            width=600, height=480
    )

    fig.update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

    if not index:
        return fig
    
    if main_type == 'original':
        x_loc = 'x'
        y_loc = 'y'
        z_loc = 'z'
    else:
        x_loc = 'x_'+version
        y_loc = 'y_'+version
        z_loc = 'z_'+version

    fig.add_trace(
        go.Scatter3d(
            x=[df.loc[index, x_loc]], 
            y=[df.loc[index, y_loc]], 
            z=[df.loc[index, z_loc]], 
            mode='markers',
            marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='MediumPurple', line=go.scatter3d.marker.Line(width=5, color='MediumPurple'))
        )
    )

    return fig


def make_latent_figure(df, version, index=False):

    main_type = version.split('_')[0]

    if main_type == "trimap":
        fig = px.scatter(
            df, x='x', y='y', color='label',
            title="TRIMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x': False, 'y': False, 'index': False, 'label': False},
            width=600, height=480, size_max=10
        )
    
    elif main_type == "umap":
        fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="UMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=600, height=480
        )
        
    elif main_type == "tsne":
        fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="T-SNE",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=600, height=480
        )

    elif main_type == "pacmap":
        fig = px.scatter(
            df, x='x_'+version, y='y_'+version, color='label',
            title="PACMAP",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'x_'+version: False, 'y_'+version: False, 'index': False, 'label': False},
            width=600, height=480
        )

    # TODO: input correct layout_dict once it's made
    fig.update_layout(fig_layout_dict_latent, showlegend=False)

    if not index:
        return fig

    # if a point is highlighted, highlight that point
    df_row = df[df['index'] == index]
    if main_type == 'trimap':
        x = float(df_row['x'])
        y = float(df_row['y'])
    else:
        x = float(df_row['x_'+version])
        y = float(df_row['y_'+version])
    
    marker = go.Scattergl(
        x=[x], y=[y],
        mode='markers',
        marker=dict(
            size=20,
            color='rgba(0, 0, 0, 1)',
            symbol='diamond-open',
            line=dict(color='rgba(0,0,0,1)', width=5)
        ),
        showlegend=False
    )

    fig.add_trace(marker)

    return fig

def get_button_name(on_latent):
    if on_latent:
        return "See Data Distribution on Embedding Space"
    return "See Data Distribution on Latent Space"