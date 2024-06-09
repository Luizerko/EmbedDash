import trimap
import numpy as np
import os
import pandas as pd

import plotly.express as px

from pathlib import Path
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, callback

from logger import logger
from models import train_and_predict, models
from utils import (
    load_mnist,
    convert_images_to_base64,
    compute_centroids,
    compute_pairwise_distances,
    objective_function,
    compute_translations,
    compute_translations,
    sort_coordinates_with_others,
    closest_points_to_line
)



####################### VARIABLES #######################


# Define the callback to update the clicked images
CLICKED_POINTS = []
CLICKED_POINTS_INDEX = 0
PREV_CLICKDATA = None

# Paths to cache files
data_path = Path("data/")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f'Folder "{data_path}" created.')
else:
    print(f'Folder "{data_path}" already exists.')
dataframe_path = data_path / "test.pkl"


app = Dash(__name__)


####################### DATA PREPROCESSING #######################


if dataframe_path.exists():
    # Load the DataFrame from the file
    df = pd.read_pickle(dataframe_path)
else:
    logger.info("Downloading MNIST data...")
    train_examples, train_labels, test_examples, test_labels = load_mnist()
    examples = np.concatenate((train_examples, test_examples), 0)
    base64_images = convert_images_to_base64(examples)
    labels = np.concatenate((train_labels, test_labels))
    indices = np.arange(len(examples))

    # Getting the latent centroids and their distances
    # In this case, since we don't have a model yet, the latent space is the data space
    # ADD A MODEL TO MAKE THE SPACE MORE INTERESTING
    examples = examples.reshape(examples.shape[0], -1)
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

    # Train models and predict labels
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model_predictions = train_and_predict(models[model_name], emb_mnist_trimap, labels, emb_mnist_trimap)
        df[model_name] = model_predictions

    optimal_positions = minimize(objective_function,
                                 np.array([*trimap_centroids.values()]).reshape(20),
                                 method='L-BFGS-B',
                                 args=(desired_distances,))
    translations = compute_translations(trimap_centroids, optimal_positions.x.reshape(10, 2))
    
    df['x_shift'] = df['label'].map(lambda label: translations[label][0])
    df['y_shift'] = df['label'].map(lambda label: translations[label][1])
    
    #df['x_shifted'] = df['x'] + x_translations
    #df['y_shifted'] = df['y'] + y_translations
    # REMOVE AS SOON AS THE CALLBACK IS WORKING
    # for i in np.unique(labels):
    #     df.loc[df['label'] == i, ['x', 'y']] += translations[i]

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_path)


####################### APP LAYOUT #######################


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
    dcc.RadioItems(options=["label", *models.keys()], 
                   value='label',
                   id='controls-and-radio-item'),
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


####################### CALLBACKS #######################


@callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Input(component_id='controls-and-radio-item', component_property='value'),
    State('scatter-plot', 'figure'),
    prevent_initial_call=True
)
def update_plot_labels(model_chosen, current_fig):
    updated_fig = px.scatter(
        df, x='x', y='y', color=model_chosen,
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
    return updated_fig


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
            df['x'] = df['x'] + df['x_shift']
            df['y'] = df['y'] + df['y_shift']

        else:
            df['x'] = df['x'] - df['x_shift']
            df['y'] = df['y'] - df['y_shift']

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


# Define the callback to update the image
@callback(
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


@callback(
    [Output('click-image', 'src'),
     Output('click-index', 'children')],
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals')]
)
def display_click_image(clickData, n_intervals):
    global CLICKED_POINTS # TODO: please don't use global variables, it's bad practice
    global CLICKED_POINTS_INDEX # TODO: please don't use global variables, it's bad practice

    if CLICKED_POINTS:
        click_image, click_label = CLICKED_POINTS[CLICKED_POINTS_INDEX][0]

        return click_image, click_label
    return '', ''


@callback(
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals')]
)
def cycle_through_clicked_points(clickData, n_intervals):
    global CLICKED_POINTS # TODO: please don't use global variables, it's bad practice
    global CLICKED_POINTS_INDEX # TODO: please don't use global variables, it's bad practice

    if CLICKED_POINTS:
        CLICKED_POINTS_INDEX = (CLICKED_POINTS_INDEX + 1) % len(CLICKED_POINTS)
    return


# Callback for gathering info on the actually clicked points
@callback(
    Output('image-reset-button', 'n_clicks'),
    [Input('scatter-plot', 'clickData'),
     Input('interval-component', 'n_intervals'),
     Input('image-reset-button', 'n_clicks')]
)
def find_clicked_points(clickData, n_intervals, n_clicks):
    global CLICKED_POINTS # TODO: please don't use global variables, it's bad practice
    global CLICKED_POINTS_INDEX # TODO: please don't use global variables, it's bad practice
    global PREV_CLICKDATA # TODO: please don't use global variables, it's bad practice

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
        PREV_CLICKDATA = clickData
    return None


if __name__ == '__main__':
    app.run_server(debug=True)
