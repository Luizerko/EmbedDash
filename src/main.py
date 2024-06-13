import trimap
import umap
import numpy as np
import pandas as pd
import os

import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pathlib import Path
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, callback

from logger import logger
from layouts import fig_layout_dict, small_fig_layout_dict, fig_layout_dict_mammoth
from models import train_and_predict, models
from utils import (
    load_mnist,
    load_mammoth,
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
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f'Folder "{data_path}" created.')
dataframe_mnist_path = data_path / "mnist_data.pkl"
dataframe_mammoth_path = data_path / "mammoth_data.pkl"

app = Dash(__name__)


####################### DATA PREPROCESSING #######################

if dataframe_mnist_path.exists():
    # Load the DataFrame from the file
    df = pd.read_pickle(dataframe_mnist_path)

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

    # Embed MNIST data for models
    emb_mnist_trimap = trimap.TRIMAP().fit_transform(examples.reshape((examples.shape[0], -1)))
    emb_mnist_umap = umap.UMAP().fit_transform(examples.reshape((examples.shape[0], -1)))

    # models that might need dimensionality reduction
    large_data = False
    #logger.info("examples:", len(examples))
    if len(examples) > 5000: # 5000 is an arbitrary choice
        large_data = True

    if large_data:
        logger.info("in PCA")
        pca = PCA(n_components=50)  # Reduce to 50 dimensions (arbitrary choice)
        examples = pca.fit_transform(examples)

    logger.info("in TSNE")
    emb_mnist_tsne = TSNE(
        n_components=2, # number of dimensions
        perplexity=30,  # balance between local and global aspect, 30 is what they used on MNIST
        n_iter=1000).fit_transform(examples.reshape((examples.shape[0], -1)))


    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x': emb_mnist_trimap[:, 0],
        'y': emb_mnist_trimap[:, 1],
        'x_umap': emb_mnist_umap[:, 0],
        'y_umap': emb_mnist_umap[:, 1],
        'x_tsne': emb_mnist_tsne[:, 0],
        'y_tsne': emb_mnist_tsne[:, 1],
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

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_mnist_path)

if dataframe_mammoth_path.exists():
    # Load the DataFrame from the file
    df_mammoth = pd.read_pickle(dataframe_mammoth_path)

else:
    df_mammoth = load_mammoth()

    logger.info("Starting embedding computations for mammoth dataset")
    
    emb_mammoth_trimap = trimap.TRIMAP(n_dims=3).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
    emb_mammoth_umap = umap.UMAP(n_components=3).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
    logger.info("Starting t-sne")
    emb_mammoth_tsne = TSNE(n_components=3, perplexity=30, n_iter=1000).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
    
    df_mammoth[['x_trimap', 'y_trimap', 'z_trimap']] = emb_mammoth_trimap
    df_mammoth[['x_umap', 'y_umap', 'z_umap']] = emb_mammoth_umap
    df_mammoth[['x_tsne', 'y_tsne', 'z_tsne']] = emb_mammoth_tsne
    
    df_mammoth.to_pickle(dataframe_mammoth_path)

########################## FIGURES ##########################


fig = px.scatter(
    df, x='x', y='y', color='label',
    title="TRIMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x': False, 'y': False, 'image': False},
    width=800, height=640, size_max=10
).update_layout(fig_layout_dict)

umap_fig = px.scatter(
    df, x='x_umap', y='y_umap', color='label',
    title="UMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_umap': False, 'y_umap': False, 'image': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)

tsne_fig = px.scatter(
    df, x='x_tsne', y='y_tsne', color='label',
    title="T-SNE Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_tsne': False, 'y_tsne': False, 'image': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)


original_mammoth = px.scatter_3d(
    df_mammoth, x='x', y='y', z='z', color='label',
    title="Original Mammoth Data",
    hover_data={'label': False, 'x': False, 'y': False, 'z': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

trimap_mammoth = px.scatter_3d(
    df_mammoth, x='x_trimap', y='y_trimap', z='z_trimap', color='label',
    title="TriMap Embedding",
    hover_data={'label': False, 'x_trimap': False, 'y_trimap': False, 'z_trimap': False},
    width=700, height=520
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

umap_mammoth = px.scatter_3d(
    df_mammoth, x='x_umap', y='y_umap', z='z_umap', color='label',
    title="UMAP Embedding",
    hover_data={'label': False, 'x_umap': False, 'y_umap': False, 'z_umap': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

tsne_mammoth = px.scatter_3d(
    df_mammoth, x='x_tsne', y='y_tsne', z='z_tsne', color='label',
    title="t-SNE Embedding",
    hover_data={'label': False, 'x_tsne': False, 'y_tsne': False, 'z_tsne': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))


####################### APP LAYOUT #######################


# fig_sub1 = px.scatter(px.data.iris(), x='petal_length', y='petal_width', color='species').update_layout(small_fig_layout_dict)
# fig_sub2 = px.scatter(px.data.iris(), x='petal_length', y='petal_width', color='species').update_layout(small_fig_layout_dict)


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='MNIST Data', children=[
            html.Div([
                ### Left side of the layout
                html.Div([
                    dcc.Graph(
                        id='scatter-plot',
                        figure=fig,
                        style={"height": "60%"}
                    ),
                ], style={'flex': '3', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justifyContent': 'flex-start', 'align-items': 'center'}),

                ### Middle of the layout
                html.Div([
                    # Box for Plots
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='UMAP-plot',
                                figure=umap_fig,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='T-SNE-plot',
                                figure=tsne_fig,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80%', 'minWidth': '230px', 'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'column'}),
                ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justify-content': 'flex-start', 'align-items': 'center'}),

                ### Right side of the layout
                html.Div([
                    # Box for RadioItems
                    html.Div([
                        html.H3("Labels", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '10px', 'margin-bottom': '5px'}),
                        dcc.RadioItems(
                            options=["Label", *models.keys()],
                            value='Label',
                            id='controls-and-radio-item',
                            labelStyle={'display': 'block', 'font-family': 'Arial'}
                        )
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin-bottom': '10px', 'width': '85%'}),

                    # Box for images
                    html.Div([
                        html.H3("Hover Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        html.Div([
                            html.Img(id='hover-image', style={'height': '200px'}),
                            html.Div(id='hover-index', style={'font-family': 'Arial', 'padding': '10px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                        html.H3("Click Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        html.Div([
                            html.Img(id='click-image', style={'height': '200px'}),
                            html.Div(id='click-index', style={'font-family': 'Arial', 'padding': '10px'})
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '85%', 'height': '100%'}),

                    html.Div([
                        html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'alignItems': 'center', 'display': 'flex', 'justifyContent': 'center', 'width': '85%', 'margin': '10px'}),

                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'margin': '10px', 'height': '80vh', 'justify-content': 'flex-start', 'align-items': 'center'}),

            ], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '100vh'})
        ]),

        dcc.Tab(label='Mammoth Data', children=[
            html.Div([

                ### Left side of the layout
                html.Div([
                    html.Div([
                        dcc.Graph(
                        id='original-mammoth',
                        figure=original_mammoth,
                        style={"height": "30%"}
                        )
                    ]),
                    html.Div([
                        dcc.Graph(
                            id='trimap-mammoth',
                            figure=trimap_mammoth,
                            style={"height": "50%"}
                        )
                    ])
                ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justify-content': 'space-between', 'align-items': 'center'}),

                ### Right of the layout
                html.Div([
                    # Box for Plots
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='UMAP Plot',
                                figure=umap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='t-SNE Plot',
                                figure=tsne_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80%', 'minWidth': '250px', 'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'column'}),

                ], style={'flex': '1', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justify-content': 'flex-start', 'align-items': 'center'}),
                
                ], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '100vh', 'flex': '0 0 auto'})
        ]),
    ])
])


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
        updated_fig.update_layout(fig_layout_dict)
        return updated_fig, n_clicks
    return current_fig, n_clicks


@callback(
    [Output('hover-image', 'src'),
     Output('hover-index', 'children'),
     Output('scatter-plot', 'hoverData'),
     Output('UMAP-plot', 'hoverData'),
     Output('T-SNE-plot', 'hoverData')],
    [Input('scatter-plot', 'hoverData'),
     Input('UMAP-plot', 'hoverData'),
     Input('T-SNE-plot', 'hoverData')]
)
def display_hover_image(MainhoverData, UMAPhoverData, TSNEhoverData):
    
    # if you are hovering over any of the input images, get that hoverData
    hoverData = None
    inputs = [MainhoverData, UMAPhoverData, TSNEhoverData]
    for inp in inputs:
        if inp is not None:
            hoverData = inp
            break
    
    if hoverData is None:
        return '', '', None, None, None

    original_label = hoverData['points'][0]['customdata'][0]
    original_image = hoverData['points'][0]['customdata'][1]

    return original_image, f'Label: {original_label}', None, None, None


@callback(
    [Output('click-image', 'src'),
     Output('click-index', 'children'),
     Output('scatter-plot', 'clickData'),
     Output('UMAP-plot', 'clickData'),
     Output('T-SNE-plot', 'clickData')],
    [Input('scatter-plot', 'clickData'),
     Input('UMAP-plot', 'clickData'),
     Input('T-SNE-plot', 'clickData')]
)
def display_click_image(MainclickData, UMAPclickData, TSNEclickData):
    clickData = None
    inputs = [MainclickData, UMAPclickData, TSNEclickData]
    for inp in inputs:
        if inp is not None:
            clickData = inp
            break

    if clickData is not None:
        original_label = clickData['points'][0]['customdata'][0]
        original_image = clickData['points'][0]['customdata'][1]

        return original_image, f'Label: {original_label}', None, None, None
    return '', '', None, None, None


if __name__ == '__main__':
    app.run_server(debug=True)
