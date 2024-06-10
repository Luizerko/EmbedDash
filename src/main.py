import trimap
import numpy as np
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
    compute_translations
)



####################### VARIABLES #######################

# Paths to cache files
data_path = Path("data/")
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
    
    # REMOVE AS SOON AS THE CALLBACK IS WORKING
    # for i in np.unique(labels):
    #     df.loc[df['label'] == i, ['x', 'y']] += translations[i]

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_path)




########################## FIGURE ##########################


fig_layout_dict = {
        "title": {
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
    "margin": dict(l=20, r=20, t=100, b=20),
    "paper_bgcolor": "White",
    "xaxis": dict(showgrid=False, zeroline=False, visible=False),
    "yaxis": dict(showgrid=False, zeroline=False, visible=False),
    "legend": dict(
        title="Label",
        traceorder="normal",
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        bgcolor="White",
    )
}


fig = px.scatter(
    df, x='x', y='y', color='label',
    title="TRIMAP embeddings on MNIST",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': True, 'x': False, 'y': False, 'image': 'image'},
    width=1000, height=800
).update_layout(fig_layout_dict)


####################### APP LAYOUT #######################


app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig,
            style={"height": "70%"}
        ),
    ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '90vh'}),

    # Right side of the layout
    html.Div([
        # Box for RadioItems
        html.Div([
            html.H4("Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
            dcc.RadioItems(
                options=["label", *models.keys()],
                value='label',
                id='controls-and-radio-item',
                labelStyle={'display': 'block', 'font-family': 'Arial'}
            )
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px'}),

        # Box for images
        html.Div([
            html.H4("Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
            html.Div([
                html.Img(id='hover-image', style={'height': '200px'}),
                html.Div(id='hover-index', style={'font-family': 'Arial', 'padding': '10px'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

            html.Div([
                html.Img(id='click-image', style={'height': '200px'}),
                html.Div(id='click-index', style={'font-family': 'Arial', 'padding': '10px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '480px', 'gap': '20px'}),

        html.Div([
            html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0)
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'alignItems': 'center', 'display': 'flex', 'justifyContent': 'center', 'minWidth': '230px'}),

    ], style={'flex': '1', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'margin': '10px', 'margin-top': '0px', 'borderRadius': '15px'}),
], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '100vh'})



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
    updated_fig.update_layout(fig_layout_dict)
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
        updated_fig.update_layout(fig_layout_dict)
        return updated_fig, n_clicks
    return current_fig, n_clicks


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
    [Input('scatter-plot', 'clickData')]
)
def display_click_image(clickData):
    if clickData is not None:
        original_label = clickData['points'][0]['customdata'][0]
        original_image = clickData['points'][0]['customdata'][1]
        x_coord = clickData['points'][0]['x']
        y_coord = clickData['points'][0]['y']

        return original_image, f'Original Label: {original_label}'
    return '', ''


if __name__ == '__main__':
    app.run_server(debug=True)
