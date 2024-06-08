import time
import base64
import io
import trimap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances
from keras.datasets import mnist
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, no_update


# Paths to cache files
#dataframe_path = 'mnist_trimap_dataframe.pkl'
data_path = Path("data/")
dataframe_path = data_path / "test.pkl"
start_time = time.time()


app = Dash(__name__)

#######################################
# HELPER FUNCTIONS
#######################################

def load_mnist(percentage=100):
    """
    outputs the mnist needed to train trimap
    used percentage for debugging and not needing to reload the entire time
    """
    (train_examples, train_labels), (test_examples, test_labels) = mnist.load_data()
    
    # Calculate the number of samples to load
    train_samples = int(len(train_examples) * percentage / 100)
    test_samples = int(len(test_examples) * percentage / 100)
    
    # Slice the arrays to get the specified percentage of data
    train_examples = train_examples[:train_samples].astype(np.float32)
    train_labels = train_labels[:train_samples]
    test_examples = test_examples[:test_samples].astype(np.float32)
    test_labels = test_labels[:test_samples]
    
    return train_examples, train_labels, test_examples, test_labels

def convert_image_to_base64(img):
    buf = io.BytesIO()
    plt.imsave(buf, img, format='png', cmap='gray')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def convert_images_to_base64(images):
    base64_images = np.array([convert_image_to_base64(img) for img in images])
    return base64_images

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
def objective_function(centroids):
    
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


# Check if the DataFrame cache exists
if dataframe_path.exists():
    # Load the DataFrame from the file
    df = pd.read_pickle(dataframe_path)
else:
    # Load MNIST dataset (e.g., 10% of the data)positions
    train_examples, train_labels, test_examples, test_labels = load_mnist()
    examples = np.concatenate((train_examples, test_examples), 0)
    base64_images = convert_images_to_base64(examples)
    labels = np.concatenate((train_labels, test_labels))
    indices = np.arange(len(examples))

    # Getting the latent centroids and their distances
    # In this case, since we don't have a model yet, the latent space is the data space
    # ADD A MODEL TO MAKE THE SPACE MORE INTERESTING
    examples = examples.reshape(examples.shape[0], examples.shape[1]*examples.shape[2])
    latent_centroids = compute_centroids(examples, labels)
    desired_distances = compute_pairwise_distances(np.array([*latent_centroids.values()]))

    # Embed MNIST data using TRIMAP
    emb_mnist_trimap = trimap.TRIMAP().fit_transform(examples.reshape((examples.shape[0], -1)))

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x': emb_mnist_trimap[:, 0],
        'y': emb_mnist_trimap[:, 1],
        'label': labels,
        'image': base64_images,
        'index': indices
    })

    # Getting the embedding centroids and computing the correspondent translation to get move them to the latent space distance
    df['x'] = (df['x'] - df['x'].min())/(df['x'].max() - df['x'].min())
    df['y'] = (df['y'] - df['y'].min())/(df['y'].max() - df['y'].min())
    trimap_centroids = compute_centroids(np.array(df[['x', 'y']]), np.array(df['label']))
    
    # import ipdb
    # ipdb.set_trace()

    optimal_positions = minimize(objective_function, np.array([*trimap_centroids.values()]).reshape(20), method='L-BFGS-B')
    translations = compute_translations(trimap_centroids, optimal_positions.x.reshape(10, 2))
    
    # REMOVE AS SOON AS THE CALLBACK IS WORKING
    # for i in np.unique(labels):
    #     df.loc[df['label'] == i, ['x', 'y']] += translations[i]

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_path)

# Create Plotly figure with customized hover data
fig = px.scatter(
    df, x='x', y='y', color='label',
    title="TRIMAP embeddings on MNIST",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': True, 'x': False, 'y': False, 'image': 'image'},
    width=1000, height=800
)

# Define the layout
app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure=fig,
        style={"padding": "10px"}
    ),
    html.Div([
        html.Div([
            html.Img(id='hover-image', style={'height': '200px'}),
            html.Div(id='hover-index'),
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
        html.Div([
            html.Img(id='click-image', style={'height': '200px'}),
            html.Div(id='click-index'),
            html.Button('Reset', id='image-reset-button', n_clicks=0),
            html.Button('Find between', id='image-find-between-button', n_clicks=0)
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
        dcc.Interval(
            id='interval-component',
            interval=500,  # Update every .5 seconds
            n_intervals=0)
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around', 'width': '100%'}),
    html.Div([
        html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0)
    ], style={'margin': '20px'})
], style={"display": "flex", "flexDirection": "column", "alignItems": "center"})


fig.update_layout(
    title={
        'text': "TRIMAP embeddings on MNIST",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 32,
            'color': 'black',
            'family': 'Arial Black'
        }
    },
    margin=dict(l=20, r=20, t=100, b=20),
    paper_bgcolor="AliceBlue",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    legend=dict(
        title="Label",
        traceorder="normal",
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        bgcolor="AliceBlue",
        bordercolor="Black",
        borderwidth=2
    )
)

# Define the callback to update the image
@app.callback(
    [Output('hover-image', 'src'),
     Output('hover-index', 'children')],
    [Input('scatter-plot', 'hoverData')]
)
def display_hover_image(hoverData):
    if hoverData is None:
        return '', ''
    original_label = hoverData['points'][0]['customdata'][0]
    original_image = hoverData['points'][0]['customdata'][1]
    return original_image, f'Original Label: {original_label}'

# Define the callback to update the clicked images
CLICKED_POINTS = []
CLICKED_POINTS_INDEX = 0
PREV_CLICKDATA = None

@app.callback(
    [Output('click-image', 'src'),
     Output('click-index', 'children')],
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals')]
)
def display_click_image(clickData, n_intervals):
    global CLICKED_POINTS
    global CLICKED_POINTS_INDEX

    if CLICKED_POINTS:
        click_image, click_label = CLICKED_POINTS[CLICKED_POINTS_INDEX][0]

        return click_image, click_label
    
    return '', ''



@app.callback(
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals')]
)
def cycle_through_clicked_points(clickData, n_intervals):
    global CLICKED_POINTS
    global CLICKED_POINTS_INDEX

    if CLICKED_POINTS:
        CLICKED_POINTS_INDEX = (CLICKED_POINTS_INDEX + 1) % len(CLICKED_POINTS)
    
    return

# callback for gathering info on the actually clicked points
@app.callback(
    Output('image-reset-button', 'n_clicks'),
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals'),
     Input('image-reset-button', 'n_clicks')]
)
def find_clicked_points(clickData, n_intervals, n_clicks):
    global CLICKED_POINTS
    global CLICKED_POINTS_INDEX
    global PREV_CLICKDATA

    if n_clicks:
        CLICKED_POINTS = []
        CLICKED_POINTS_INDEX = 0
        # Return an empty string to clear the image
        return None


    if (clickData is not None) and not (clickData == PREV_CLICKDATA):
        original_label = clickData['points'][0]['customdata'][0]
        original_image = clickData['points'][0]['customdata'][1]
        x_coord = clickData['points'][0]['x']
        y_coord = clickData['points'][0]['y']
        CLICKED_POINTS.append([(original_image, f'Original Label: {original_label}'), (x_coord, y_coord)])

        # reset clickData until you click on something again
        PREV_CLICKDATA=clickData
    
    return None

# update CLICKED_POINTS to finde the points in between
def fit_line(coordinates):
    x_coords, y_coords = zip(*coordinates)

    if len(coordinates) == 2:
        poly = np.polyfit(x_coords, y_coords, 1)
        return 0, poly[0], poly[1]
    else:
        poly = np.polyfit(x_coords, y_coords, 2)
        return poly[0], poly[1], poly[2]


from scipy.optimize import minimize
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

@app.callback(
    Output('image-find-between-button', 'n_clicks'),
    Input('image-find-between-button', 'n_clicks')
)
def find_images_between_clicks(n_clicks):
    if not n_clicks:
        return None

    global CLICKED_POINTS
    global CLICKED_POINTS_INDEX


    coords = []
    for _, coord in CLICKED_POINTS:
        coords.append(coord)
    
    xs = np.array([coord[0] for coord in coords])
    min_x = np.min(xs)
    max_x = np.max(xs)
    ys = np.array([coord[1] for coord in coords])
    min_y = np.min(ys)
    max_y = np.max(ys)


    global fig
    labels = []
    imgs = []
    x_coords = []
    y_coords = []
    for trace in fig.data:
        for x_coord in trace.x:
            x_coords.append(x_coord)
        for y_coord in trace.y:
            y_coords.append(y_coord)

        for data in trace.customdata:
            labels.append(data[0])
            imgs.append(data[1])

    all_coords = []
    sublabels = []
    subimgs = []
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if (x >= min_x) and (x <= max_x) and (y >= min_y) and (y <= max_y):
            all_coords.append((x,y))
            sublabels.append(labels[i])
            subimgs.append(imgs[i])

    sorted_coords, (sorted_imgs, sorted_labes) = sort_coordinates_with_others(all_coords, subimgs, sublabels)
  
    CLICKED_POINTS = closest_points_to_line(sorted_coords, sorted_imgs, sorted_labes, 1)
    CLICKED_POINTS_INDEX = 0
    
    return None



# Callback for the latent space distances function
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('translate-button', 'n_clicks')],
    [Input('translate-button', 'n_clicks')],
    [State('scatter-plot', 'figure')]
)
def update_plot(n_clicks, current_fig):

    # instead of this n_clicks%2 stuff, you can return n_clicks, allowing you to reset it to 0
    # tip from Joost
    if n_clicks > 0:

        if n_clicks%2 != 0:
            for i in np.unique(df['label']):
                df.loc[df['label'] == i, ['x', 'y']] += translations[i]
        else:
            for i in np.unique(df['label']):
                df.loc[df['label'] == i, ['x', 'y']] -= translations[i]

        updated_fig = px.scatter(
            df, x='x', y='y', color='label',
            title="TRIMAP embeddings on MNIST",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': True, 'x': False, 'y': False, 'image': False},
            width=1000, height=800
        )
        updated_fig.update_layout(
            title={
                'text': "TRIMAP embeddings on MNIST",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 32,
                    'color': 'black',
                    'family': 'Arial Black'
                }
            },
            margin=dict(l=20, r=20, t=100, b=20),
            paper_bgcolor="AliceBlue",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            legend=dict(
                title="Label",
                traceorder="normal",
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
                bgcolor="AliceBlue",
                bordercolor="Black",
                borderwidth=2
            )
        )
        return updated_fig, n_clicks
    
    return current_fig, n_clicks

if __name__ == '__main__':
    app.run_server(debug=True)
