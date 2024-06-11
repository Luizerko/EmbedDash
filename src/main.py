import trimap
import umap
import numpy as np
import pandas as pd

import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pathlib import Path
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, callback

from logger import logger
from layouts import fig_layout_dict, small_fig_layout_dict
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

    # Embed MNIST data for models
    emb_mnist_trimap = trimap.TRIMAP().fit_transform(examples.reshape((examples.shape[0], -1)))
    emb_mnist_umap = umap.UMAP().fit_transform(examples.reshape((examples.shape[0], -1)))

    # models that might need dimensionality reduction
    large_data = False
    logger.info("examples:", len(examples))
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
    
    # REMOVE AS SOON AS THE CALLBACK IS WORKING
    # for i in np.unique(labels):
    #     df.loc[df['label'] == i, ['x', 'y']] += translations[i]

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_path)




########################## FIGURES ##########################


fig = px.scatter(
    df, x='x', y='y', color='label',
    title="TRIMAP embeddings on MNIST",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': True, 'x': False, 'y': False, 'image': 'image'},
    width=800, height=640
).update_layout(fig_layout_dict)

umap_fig = px.scatter(
    df, x='x_umap', y='y_umap', color='label',
    title="UMAP embeddings on MNIST",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': True, 'x_umap': False, 'y_umap': False, 'image': 'image'},
    width=400, height=320
).update_layout(small_fig_layout_dict)

tsne_fig = px.scatter(
    df, x='x_tsne', y='y_tsne', color='label',
    title="T-SNE embeddings on MNIST",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': True, 'x_tsne': False, 'y_tsne': False, 'image': 'image'},
    width=400, height=320
).update_layout(small_fig_layout_dict)


####################### APP LAYOUT #######################


# fig_sub1 = px.scatter(px.data.iris(), x='petal_length', y='petal_width', color='species').update_layout(small_fig_layout_dict)
# fig_sub2 = px.scatter(px.data.iris(), x='petal_length', y='petal_width', color='species').update_layout(small_fig_layout_dict)



app.layout = html.Div([
    ### Left side of the layout
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig,
            style={"height": "60%"}
        ),
    ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '90vh'}),



    ### Middle of the layout
    html.Div([
        # Box for Plots
        html.Div([
            html.H3("Other methods", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '30px', 'margin-bottom': '5px'}),
            html.Div([
                html.H4("UMAP", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                dcc.Graph(
                    id='UMAP-plot',
                    figure=umap_fig,
                    style={"width": "100%", "display": "inline-block", 'height': '300px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

            html.Div([
                html.H4("t-SNE", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '30px', 'margin-bottom': '5px'}),
                dcc.Graph(
                    id='T-SNE-plot',
                    figure=tsne_fig,
                    style={"width": "100%", "display": "inline-block", 'height': '300px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '480px', 'minWidth': '230px'}),
    ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '90vh'}),


    ### Right side of the layout
    html.Div([
        # Box for RadioItems
        html.Div([
            html.H3("Labels", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '30px', 'margin-bottom': '5px'}),
            dcc.RadioItems(
                options=["label", *models.keys()],
                value='label',
                id='controls-and-radio-item',
                labelStyle={'display': 'block', 'font-family': 'Arial'}
            )
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px'}),

        # Box for images
        html.Div([
            html.H3("Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
            html.Div([
                html.Img(id='hover-image', style={'height': '200px'}),
                html.Div(id='hover-index', style={'font-family': 'Arial', 'padding': '10px'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

            html.Div([
                html.Img(id='click-image', style={'height': '200px'}),
                html.Div(id='click-index', style={'font-family': 'Arial', 'padding': '10px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '500px', 'minWidth': '230px'}),

        html.Div([
            html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0)
        ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'alignItems': 'center', 'display': 'flex', 'justifyContent': 'center', 'minWidth': '230px'}),

    ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'margin': '10px'}),
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

    return original_image, f'Original Label: {original_label}', None, None, None


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

        return original_image, f'Original Label: {original_label}', None, None, None
    return '', '', None, None, None


if __name__ == '__main__':
    app.run_server(debug=True)
