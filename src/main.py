import trimap
import numpy as np
import pandas as pd

import callbacks
import plotly.express as px

from pathlib import Path
from scipy.optimize import minimize
from dash import Dash, html, dcc

from utils import (
    load_mnist,
    convert_images_to_base64,
    compute_centroids,
    compute_pairwise_distances,
    objective_function,
    compute_translations,
    compute_translations
)


# Define the callback to update the clicked images
CLICKED_POINTS = []
CLICKED_POINTS_INDEX = 0
PREV_CLICKDATA = None

# Paths to cache files
data_path = Path("data/")
dataframe_path = data_path / "test.pkl"


app = Dash(__name__)


### MAIN ###


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

    optimal_positions = minimize(objective_function, 
                                 np.array([*trimap_centroids.values()]).reshape(20), 
                                 method='L-BFGS-B',
                                 args=(desired_distances,)).x
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

if __name__ == '__main__':
    app.run_server(debug=True)
