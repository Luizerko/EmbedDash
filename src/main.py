import trimap
import umap
import pacmap
import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pathlib import Path
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from logger import logger
from layouts import fig_layout_dict, small_fig_layout_dict, fig_layout_dict_mammoth
from models import train_and_predict, generate_latent_data, models

from utils import make_mnist_figure


####################### VARIABLES #######################

# Paths to cache files
data_path = Path("data/")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f'Folder "{data_path}" created.')
dataframe_mnist_path = data_path / "mnist_param_grid_data.pkl"
dataframe_mammoth_path = data_path / "mammoth_param_grid_data.pkl"
latent_data_path = data_path / "latent_data.pkl"

app = Dash(__name__)


####################### DATA PREPROCESSING #######################


### MNIST Data
if dataframe_mnist_path.exists():
    # Load the DataFrame from the file
    df_mnist = pd.read_pickle(dataframe_mnist_path)

### Mammoth Data
if dataframe_mammoth_path.exists():
    # Load the DataFrame from the file
    df_mammoth = pd.read_pickle(dataframe_mammoth_path)

### Latent Data
if not latent_data_path.exists():
    df_latent = generate_latent_data()
    df_latent.to_pickle(latent_data_path)
else:
    # Load the latent data from the file
    df_latent = pd.read_pickle(latent_data_path)

mnist_plot_dictionary = {'main': 'trimap', 'subplot_1': 'umap', 'subplot_2': 'tsne', 'subplot_3': 'pacmap'}
mnist_embedding_dictionary = {'trimap': 'main', 'umap': 'subplot_1', 'tsne': 'subplot_2', 'pacmap': 'subplot_3'}

########################## FIGURES ##########################


### MNIST Data
trimap_mnist = px.scatter(
                    df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                    title="TRIMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False, 'index': False},
                    width=800, height=640, size_max=10
                ).update_layout(fig_layout_dict)

umap_mnist = px.scatter(
                    df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                    title="UMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False, 'index': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)

tsne_mnist = px.scatter(
                    df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                    title="T-SNE Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False, 'index': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)

pacmap_mnist = px.scatter(
                    df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                    title="PaCMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False, 'index': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)


### Mammoth Data
original_mammoth = px.scatter_3d(
    df_mammoth, x='x', y='y', z='z', color='label',
    title="Original Mammoth Data",
    hover_data={'label': False, 'x': False, 'y': False, 'z': False},
    width=700, height=520
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

trimap_mammoth = px.scatter_3d(
    df_mammoth, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', z='z_trimap_nin_12_nout_4', color='label',
    title="TriMap Embedding",
    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'z_trimap_nin_12_nout_4': False},
    width=700, height=520
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

umap_mammoth = px.scatter_3d(
    df_mammoth, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', z='z_umap_nneighbors_15_mindist_0.1', color='label',
    title="UMAP Embedding",
    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'z_umap_nneighbors_15_mindist_0.1': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

tsne_mammoth = px.scatter_3d(
    df_mammoth, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', z='z_tsne_perp_30_exa_12', color='label',
    title="t-SNE Embedding",
    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'z_tsne_perp_30_exa_12': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

pacmap_mammoth = px.scatter_3d(
    df_mammoth, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', z='z_pacmap_nneighbors_10_init_pca', color='label',
    title="PaCMAP Embedding",
    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'z_pacmap_nneighbors_10_init_pca': False},
    width=500, height=420
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))


### Latent Data
fig_latent = px.scatter(
    df_latent, x='x', y='y', color='label',
    title="TRIMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x': False, 'y': False},
    width=800, height=640, size_max=10
).update_layout(fig_layout_dict)

umap_fig_latent = px.scatter(
    df_latent, x='x_umap', y='y_umap', color='label',
    title="UMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_umap': False, 'y_umap': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)

tsne_fig_latent = px.scatter(
    df_latent, x='x_tsne', y='y_tsne', color='label',
    title="T-SNE Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_tsne': False, 'y_tsne': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)

####################### APP LAYOUT #######################

multiplicative_values = [2**i for i in range(1, 4)]
marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='MNIST Data', children=[
            html.Div([
                ### Upper row
                html.Div([
                    html.Div([
                        dcc.Store(id='version_parameters', storage_type='local', data=['trimap_nin_12_nout_4', 'umap_nneighbors_15_mindist_0.1', 'tsne_perp_30_exa_12', 'pacmap_nneighbors_10_init_pca']),
                        dcc.Graph(
                            id='mnist_main_plot', 
                            figure=trimap_mnist,
                            style={}
                        ),
                        html.Div([
                            html.Label('Number of Inliers:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                            dcc.Slider(min=8, max=16, value=12, step=4,
                            id='mnist_slider_1',
                            ),
                            html.Label('Number of Outliers:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                            dcc.Slider(min=0, max=len(multiplicative_values)-1, value=1, step=None, marks=marks,
                            id='mnist_slider_2',
                            ),
                            html.Div([], style={'height': '50px'}),
                            html.Label('Choose the Embedding:', style={}), 
                            dcc.Dropdown(
                                options=['trimap', 'umap', 'tsne', 'pacmap'], value='trimap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
                            ),
                        ], id='sliders_div', style={'width': '500px', 'height': '300px'}),
                    ], style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'border-radius': '15px', 'margin': '10px', 'justify-content': 'space-around', 'align-items': 'center', 'background': '#FFFFFF', 'width': '95%'}),

                    ### Lower row
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='mnist_subplot_1', 
                                figure=umap_mnist,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='mnist_subplot_2',
                                figure=tsne_mnist,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='mnist_subplot_3',
                                figure=pacmap_mnist,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

                    ], style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'border-radius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '95%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),

                ], style={'display': 'flex', 'padding': '20px', 'flex-direction': 'row', 'flex-wrap': 'wrap', 'width': '80vw'}),

                ### Right column
                html.Div([

                    # Box for images
                    html.Div([
                        html.H3("Hover Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            html.Img(id='hover-image', style={'height': '200px'}),
                            html.Div(id='hover-index', style={'font-family': 'Arial', 'padding': '10px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                        
                        html.H3("Click Sample", style={'text-align': 'center', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            dcc.Store(id='clicked-index', storage_type='local', data=None),
                            html.Img(id='click-image', style={'height': '200px'}),
                            html.Div(id='click-index', style={'font-family': 'Arial', 'padding': '10px'})
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
                    ], style={'padding': '20px', 'border-radius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '85%'}),

                    html.Div([
                        html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),

                ], style={'display': 'flex', 'flex-direction': 'column', 'border-radius': '15px', 'margin': '10px', 'align-items': 'center', 'width': '15vw', 'justify-content': 'center'}),

            ], style={"display": "flex", 'flex-wrap': 'nowrap', "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '90vh'})
        ]),

        dcc.Tab(label='Mammoth Data', children=[
            html.Div([
                #Upper row
                html.Div([
                    html.Div([
                        dcc.Graph(
                        id='mammoth_original_plot',
                        figure=original_mammoth,
                        style={}
                        )
                    ]),
                    html.Div([
                        dcc.Graph(
                            id='mammoth_trimap_plot',
                            figure=trimap_mammoth,
                            style={}
                        )
                    ]),
                    html.Div([
                        html.Button('PARAMETERS GO HERE', id='placeholdedr-button2', n_clicks=0)
                    ]),

                ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '95%', 'justify-content': 'space-around', 'align-items': 'center'}),

                #Lower row
                html.Div([
                        html.Div([
                            dcc.Graph(
                                id='mammoth_umap_plot',
                                figure=umap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_tsne_plot',
                                figure=tsne_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_pacmap_plot',
                                figure=pacmap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '95%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),
                
                ], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'flex-wrap': 'wrap', 'height': '90vh'})
        ]),
        
        ### Latent Data
        dcc.Tab(label='Latent Data', children=[
            html.Div([
                ### Left side of the layout
                html.Div([
                    dcc.Graph(
                        id='scatter-plot-latent',
                        figure=fig_latent,
                        style={"height": "60%"}
                    ),
                ], style={'flex': '3', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justifyContent': 'flex-start', 'align-items': 'center'}),

                ### Middle of the layout
                html.Div([
                    # Box for Plots
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='UMAP-plot-latent',
                                figure=umap_fig_latent,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='T-SNE-plot-latent',
                                figure=tsne_fig_latent,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                    ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80%', 'minWidth': '230px', 'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'column'}),
                ], style={'flex': '2', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'height': '80vh', 'justify-content': 'flex-start', 'align-items': 'center'}),
            ], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '100vh'})
        ]),
    ])
])


####################### CALLBACKS #######################

# Callback for the latent space distances function
@app.callback(
    [Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('translate-button', 'n_clicks')],
    [Input('translate-button', 'n_clicks')],
    [State('mnist_main_plot', 'figure'),
     State('mnist_dd_choose_embedding', 'value')],
     prevent_initial_call=True
)
def update_plot(n_clicks, current_figure, dropdown_value):

    # instead of this n_clicks%2 stuff, you can return n_clicks, allowing you to reset it to 0
    # tip from Joost
    if dropdown_value == 'trimap':
        if n_clicks > 0:

            if n_clicks%2 != 0:
                df_mnist['x_trimap_nin_12_nout_4'] = df_mnist['x_trimap_nin_12_nout_4'] + df_mnist['x_trimap_shift']
                df_mnist['y_trimap_nin_12_nout_4'] = df_mnist['y_trimap_nin_12_nout_4'] + df_mnist['y_trimap_shift']
            else:
                df_mnist['x_trimap_nin_12_nout_4'] = df_mnist['x_trimap_nin_12_nout_4'] - df_mnist['x_trimap_shift']
                df_mnist['y_trimap_nin_12_nout_4'] = df_mnist['y_trimap_nin_12_nout_4'] - df_mnist['y_trimap_shift']

            updated_fig = px.scatter(
                df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                title="TRIMAP Embedding ",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False, 'index': False},
                width=800, height=640, size_max=10
            )
            updated_fig.update_layout(fig_layout_dict)
            return updated_fig, n_clicks

    elif dropdown_value == 'umap':
        if n_clicks > 0:

            if n_clicks%2 != 0:
                df_mnist['x_umap_nneighbors_15_mindist_0.1'] = df_mnist['x_umap_nneighbors_15_mindist_0.1'] + df_mnist['x_umap_shift']
                df_mnist['y_umap_nneighbors_15_mindist_0.1'] = df_mnist['y_umap_nneighbors_15_mindist_0.1'] + df_mnist['y_umap_shift']
            else:
                df_mnist['x_umap_nneighbors_15_mindist_0.1'] = df_mnist['x_umap_nneighbors_15_mindist_0.1'] - df_mnist['x_umap_shift']
                df_mnist['y_umap_nneighbors_15_mindist_0.1'] = df_mnist['y_umap_nneighbors_15_mindist_0.1'] - df_mnist['y_umap_shift']

            updated_fig = px.scatter(
                df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                title="UMAP Embedding ",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False, 'index': False},
                width=800, height=640, size_max=10
            )
            updated_fig.update_layout(fig_layout_dict)
            return updated_fig, n_clicks

    elif dropdown_value == 'tsne':
        if n_clicks > 0:

            if n_clicks%2 != 0:
                df_mnist['x_tsne_perp_30_exa_12'] = df_mnist['x_tsne_perp_30_exa_12'] + df_mnist['x_tsne_shift']
                df_mnist['y_tsne_perp_30_exa_12'] = df_mnist['y_tsne_perp_30_exa_12'] + df_mnist['y_tsne_shift']
            else:
                df_mnist['x_tsne_perp_30_exa_12'] = df_mnist['x_tsne_perp_30_exa_12'] - df_mnist['x_tsne_shift']
                df_mnist['y_tsne_perp_30_exa_12'] = df_mnist['y_tsne_perp_30_exa_12'] - df_mnist['y_tsne_shift']

            updated_fig = px.scatter(
                df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                title="TSNE Embedding",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False, 'index': False},
                width=800, height=640, size_max=10
            )
            updated_fig.update_layout(fig_layout_dict)
            return updated_fig, n_clicks
        
    elif dropdown_value == 'pacmap':
        if n_clicks > 0:

            if n_clicks%2 != 0:
                df_mnist['x_pacmap_nneighbors_10_init_pca'] = df_mnist['x_pacmap_nneighbors_10_init_pca'] + df_mnist['x_pacmap_shift']
                df_mnist['y_pacmap_nneighbors_10_init_pca'] = df_mnist['y_pacmap_nneighbors_10_init_pca'] + df_mnist['y_pacmap_shift']
            else:
                df_mnist['x_pacmap_nneighbors_10_init_pca'] = df_mnist['x_pacmap_nneighbors_10_init_pca'] - df_mnist['x_pacmap_shift']
                df_mnist['y_pacmap_nneighbors_10_init_pca'] = df_mnist['y_pacmap_nneighbors_10_init_pca'] - df_mnist['y_pacmap_shift']

            updated_fig = px.scatter(
                df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                title="PACMAP Embedding",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False, 'index': False},
                width=800, height=640, size_max=10
            )
            updated_fig.update_layout(fig_layout_dict)
            return updated_fig, n_clicks
    
    return current_figure, n_clicks


@app.callback(
    [Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('version_parameters', 'data', allow_duplicate=True)],
    [Input('mnist_slider_1', 'value'),
     Input('mnist_slider_2', 'value')],
    [State('version_parameters', 'data'),
     State('clicked-index', 'data')],
    prevent_initial_call=True
)
def sliders(slider_value_1, slider_value_2, version_parameters, clicked_index):
    if mnist_plot_dictionary["main"] == "trimap":
        trimap_nout_values=[2, 4, 8]
        version = 'trimap_nin_'+str(slider_value_1)+'_nout_'+str(trimap_nout_values[slider_value_2])
        version_parameters[0] = version
        updated_fig = make_mnist_figure(df_mnist, version, index=clicked_index)
        
    elif mnist_plot_dictionary["main"] == "umap":
        umap_nneighbors_values=[5, 15, 45]
        umap_mindist_values=[0.0, 0.1, 0.5]
        version = 'umap_nneighbors_'+str(umap_nneighbors_values[slider_value_1])+'_mindist_'+str(umap_mindist_values[slider_value_2])
        version_parameters[1] = version
        updated_fig = make_mnist_figure(df_mnist, version, index=clicked_index)
        
    elif mnist_plot_dictionary["main"] == "tsne":
        tsne_exa_values=[6, 12, 24]
        version = 'tsne_perp_'+str(slider_value_1)+'_exa_'+str(tsne_exa_values[slider_value_2])
        version_parameters[2] = version
        updated_fig = make_mnist_figure(df_mnist, version, index=clicked_index)

    elif mnist_plot_dictionary["main"] == "pacmap":
        pacmap_nneighbors_values=[5, 10, 20]
        pacmap_init_values=['pca', 'random']
        version = 'pacmap_nneighbors_'+str(pacmap_nneighbors_values[slider_value_1])+'_init_'+pacmap_init_values[slider_value_2]
        version_parameters[3] = version
        updated_fig = make_mnist_figure(df_mnist, version, index=clicked_index)

    return updated_fig, version_parameters

#     updated_fig.update_layout(fig_layout_dict)

#     return updated_fig

@callback(
    [Output('hover-image', 'src'),
     Output('hover-index', 'children'),
     Output('mnist_main_plot', 'hoverData'),
     Output('mnist_subplot_1', 'hoverData'),
     Output('mnist_subplot_2', 'hoverData'),
     Output('mnist_subplot_3', 'hoverData')],
    [Input('mnist_main_plot', 'hoverData'),
     Input('mnist_subplot_1', 'hoverData'),
     Input('mnist_subplot_2', 'hoverData'),
     Input('mnist_subplot_3', 'hoverData')]

)
def display_hover_image(MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData):
    
    # if you are hovering over any of the input images, get that hoverData
    hoverData = None
    inputs = [MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData]
    for inp in inputs:
        if inp is not None:
            hoverData = inp
            break
    
    if hoverData is None:
        return '', '', None, None, None, None

    original_label = hoverData['points'][0]['customdata'][0]
    original_image = hoverData['points'][0]['customdata'][1]

    return original_image, f'Label: {original_label}', None, None, None, None


@callback(
    [Output('click-image', 'src'),
     Output('click-index', 'children'),
     Output('clicked-index', 'data'),
     Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('mnist_subplot_1', 'figure', allow_duplicate=True),
     Output('mnist_subplot_2', 'figure', allow_duplicate=True),
     Output('mnist_subplot_3', 'figure', allow_duplicate=True)],
    [Input('mnist_main_plot', 'clickData'),
     Input('mnist_subplot_1', 'clickData'),
     Input('mnist_subplot_2', 'clickData'),
     Input('mnist_subplot_3', 'clickData')],
    [State('version_parameters', 'data'),
     State('mnist_dd_choose_embedding', 'value')],
    prevent_initial_call=True
)
def display_click_image(MainclickData, Sub1clickData, Sub2clickData, Sub3clickData, version_parameters, current_main):
    clickData = None
    inputs = [MainclickData, Sub1clickData, Sub2clickData, Sub3clickData]
    for inp in inputs:
        if inp is not None:
            clickData = inp
            break

    if clickData is None:
        raise PreventUpdate

    original_label = clickData['points'][0]['customdata'][0]
    original_image = clickData['points'][0]['customdata'][1]
    original_index = clickData['points'][0]['customdata'][2]

    trimap_base_version = version_parameters[0]
    umap_base_version = version_parameters[1]
    tsne_base_version = version_parameters[2]
    pacmap_base_version = version_parameters[3]

    logger.info(original_index)

    # we need to keep track of where what is exactly because it does not keep place.
    # use mnist_embedding_dictionary?
    if current_main == 'trimap':
        main_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=False)
        sub1_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
        sub2_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
        sub3_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
        
    elif current_main == 'umap':
        main_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=False)
        sub1_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=True)
        sub2_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
        sub3_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
    
    elif current_main == 'tsne':
        main_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=False)
        sub1_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=True)
        sub2_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
        sub3_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
    
    elif current_main == 'pacmap':
        main_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=False)
        sub1_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
        sub2_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
        sub3_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
    else:
        logger.info('should not be here')
        
    return original_image, f'Label: {original_label}, {version_parameters}', original_index, main_img, sub1_img, sub2_img, sub3_img


@app.callback(
    [#Output('mnist_dd_choose_embedding', 'dropdown_value'),
     Output('mnist_main_plot', 'figure'),
     Output('mnist_subplot_1', 'figure'),
     Output('mnist_subplot_2', 'figure'),
     Output('mnist_subplot_3', 'figure'), 
     Output('sliders_div', 'children')],
    [Input('mnist_dd_choose_embedding', 'value')],
    [State('version_parameters', 'data'),
     State('clicked-index', 'data')]
)
def switch_main_img(dropdown_value, version_parameters, clicked_index):
    if dropdown_value == mnist_plot_dictionary['main']:
        raise PreventUpdate


    trimap_base_version = version_parameters[0]
    umap_base_version = version_parameters[1]
    tsne_base_version = version_parameters[2]
    pacmap_base_version = version_parameters[3]

    if dropdown_value == 'trimap':
        out_main_plot = make_mnist_figure(df_mnist, trimap_base_version, index=clicked_index)

        multiplicative_values = [2**i for i in range(1, 4)]
        marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

        trimap_nin_value = int(trimap_base_version.split('_')[2])
        trimap_nout_value = trimap_base_version.split('_')[4]
        rev_marks = {str(value): i for i, value in enumerate(multiplicative_values)}

        new_slider_div = [
            html.Label('Number of Inliers:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=8, max=16, value=trimap_nin_value, step=4,
            id='mnist_slider_1',
            ),
            html.Label('Number of Outliers:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values)-1, value=rev_marks[trimap_nout_value], step=None, marks=marks,
            id='mnist_slider_2',
            ),
            html.Div([], style={'height': '50px'}),
            html.Label('Choose the Embedding:', style={}), 
            dcc.Dropdown(
                options=['trimap', 'umap', 'tsne', 'pacmap'], value='trimap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
            ),
        ]

        if mnist_plot_dictionary['main'] == 'umap':
            out_subplot = make_mnist_figure(df_mnist, umap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'umap'
            mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['trimap']
        
        elif mnist_plot_dictionary['main'] == 'tsne':
            out_subplot = make_mnist_figure(df_mnist, tsne_base_version, index=clicked_index, is_subplot=True)

            mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'tsne'
            mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['trimap']

        elif mnist_plot_dictionary['main'] == 'pacmap':
            out_subplot = make_mnist_figure(df_mnist, pacmap_base_version, index=clicked_index, is_subplot=True)

            mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['trimap']

        if mnist_embedding_dictionary['trimap'] == 'subplot_1':
            mnist_plot_dictionary['main'] = 'trimap'
            mnist_embedding_dictionary['trimap'] = 'main'
        
            return out_main_plot, out_subplot, no_update, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['trimap'] == 'subplot_2':
            mnist_plot_dictionary['main'] = 'trimap'
            mnist_embedding_dictionary['trimap'] = 'main'
        
            return out_main_plot, no_update, out_subplot, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['trimap'] == 'subplot_3':
            mnist_plot_dictionary['main'] = 'trimap'
            mnist_embedding_dictionary['trimap'] = 'main'
        
            return out_main_plot, no_update, no_update, out_subplot, new_slider_div
        
    elif dropdown_value == 'umap':
        out_main_plot = make_mnist_figure(df_mnist, umap_base_version, index=clicked_index)

        multiplicative_values_1 = [5, 15, 45]
        marks_1 = {i: str(value) for i, value in enumerate(multiplicative_values_1)}
        rev_marks_1 = {str(value): i for i, value in enumerate(multiplicative_values_1)}

        multiplicative_values_2 = [0.0, 0.1, 0.5]
        marks_2 = {i: str(value) for i, value in enumerate(multiplicative_values_2)}
        rev_marks_2 = {str(value): i for i, value in enumerate(multiplicative_values_2)}        

        umap_nneighbors_value = umap_base_version.split('_')[2]
        umap_mindist_value = umap_base_version.split('_')[4]

        new_slider_div = [
            html.Label('Number of Neighbors:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values_1)-1, value=rev_marks_1[umap_nneighbors_value], step=None, marks=marks_1,
            id='mnist_slider_1',
            ),
            html.Label('Minimum Distance:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values_2)-1, value=rev_marks_2[umap_mindist_value], step=None, marks=marks_2,
            id='mnist_slider_2',
            ),
            html.Div([], style={'height': '50px'}),
            html.Label('Choose the Embedding:', style={}), 
            dcc.Dropdown(
                options=['trimap', 'umap', 'tsne', 'pacmap'], value='umap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
            ),
        ]

        if mnist_plot_dictionary['main'] == 'trimap':
            out_subplot = make_mnist_figure(df_mnist, trimap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'trimap'
            mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['umap']
        
        elif mnist_plot_dictionary['main'] == 'tsne':
            out_subplot = make_mnist_figure(df_mnist, tsne_base_version, index=clicked_index, is_subplot=True)

            mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'tsne'
            mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['umap']

        elif mnist_plot_dictionary['main'] == 'pacmap':
            out_subplot = make_mnist_figure(df_mnist, pacmap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['umap']

        if mnist_embedding_dictionary['umap'] == 'subplot_1':
            mnist_plot_dictionary['main'] = 'umap'
            mnist_embedding_dictionary['umap'] = 'main'
        
            return out_main_plot, out_subplot, no_update, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['umap'] == 'subplot_2':
            mnist_plot_dictionary['main'] = 'umap'
            mnist_embedding_dictionary['umap'] = 'main'
        
            return out_main_plot, no_update, out_subplot, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['umap'] == 'subplot_3':
            mnist_plot_dictionary['main'] = 'umap'
            mnist_embedding_dictionary['umap'] = 'main'
        
            return out_main_plot, no_update, no_update, out_subplot, new_slider_div
        
    elif dropdown_value == 'tsne':
        out_main_plot = make_mnist_figure(df_mnist, tsne_base_version, index=clicked_index)

        # perplexity 30 min=15 max=45
        # early_exageration 12 min=6 max=24
        multiplicative_values = [6, 12, 24]
        marks = {i: str(value) for i, value in enumerate(multiplicative_values)}
        rev_marks = {str(value): i for i, value in enumerate(multiplicative_values)}


        tsne_perp_value = int(tsne_base_version.split('_')[2])
        tsne_exa_value = tsne_base_version.split('_')[4]

        new_slider_div = [
            html.Label('Perplexity:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=15, max=45, value=tsne_perp_value, step=15,
            id='mnist_slider_1',
            ),
            html.Label('Early Exagerration:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values)-1, value=rev_marks[tsne_exa_value], step=None, marks=marks,
            id='mnist_slider_2',
            ),
            html.Div([], style={'height': '50px'}),
            html.Label('Choose the Embedding:', style={}), 
            dcc.Dropdown(
                options=['trimap', 'umap', 'tsne', 'pacmap'], value='tsne', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
            ),
        ]

        if mnist_plot_dictionary['main'] == 'trimap':
            out_subplot = make_mnist_figure(df_mnist, trimap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'trimap'
            mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['tsne']
        
        elif mnist_plot_dictionary['main'] == 'umap':
            out_subplot = make_mnist_figure(df_mnist, umap_base_version, index=clicked_index, is_subplot=True)

            mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'umap'
            mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['tsne']

        elif mnist_plot_dictionary['main'] == 'pacmap':
            out_subplot = make_mnist_figure(df_mnist, pacmap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['tsne']

        if mnist_embedding_dictionary['tsne'] == 'subplot_1':
            mnist_plot_dictionary['main'] = 'tsne'
            mnist_embedding_dictionary['tsne'] = 'main'
        
            return out_main_plot, out_subplot, no_update, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['tsne'] == 'subplot_2':
            mnist_plot_dictionary['main'] = 'tsne'
            mnist_embedding_dictionary['tsne'] = 'main'
        
            return out_main_plot, no_update, out_subplot, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['tsne'] == 'subplot_3':
            mnist_plot_dictionary['main'] = 'tsne'
            mnist_embedding_dictionary['tsne'] = 'main'
        
            return out_main_plot, no_update, no_update, out_subplot, new_slider_div

    elif dropdown_value == 'pacmap':
        out_main_plot = make_mnist_figure(df_mnist, pacmap_base_version, index=clicked_index)

        multiplicative_values_1 = [5, 10, 20]
        marks_1 = {i: str(value) for i, value in enumerate(multiplicative_values_1)}
        rev_marks_1 = {str(value): i for i, value in enumerate(multiplicative_values_1)}


        multiplicative_values_2 = ['pca', 'random']
        marks_2 = {i: str(value) for i, value in enumerate(multiplicative_values_2)}
        rev_marks_2 = {str(value): i for i, value in enumerate(multiplicative_values_2)}

        pacmap_nn_value = pacmap_base_version.split('_')[2]
        pacmap_init_value = pacmap_base_version.split('_')[4]

        new_slider_div = [
            html.Label('Number of Neighbors:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values_1)-1, value=rev_marks_1[pacmap_nn_value], step=None, marks=marks_1,
            id='mnist_slider_1',
            ),
            html.Label('Initialization:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Slider(min=0, max=len(multiplicative_values_2)-1, value=rev_marks_2[pacmap_init_value], step=None, marks=marks_2,
            id='mnist_slider_2',
            ),
            html.Div([], style={'height': '50px'}),
            html.Label('Choose the Embedding:', style={}), 
            dcc.Dropdown(
                options=['trimap', 'umap', 'tsne', 'pacmap'], value='pacmap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
            ),
        ]

        if mnist_plot_dictionary['main'] == 'trimap':
            out_subplot = make_mnist_figure(df_mnist, trimap_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'trimap'
            mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['pacmap']
        
        elif mnist_plot_dictionary['main'] == 'umap':
            out_subplot = make_mnist_figure(df_mnist, umap_base_version, index=clicked_index, is_subplot=True)

            mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'umap'
            mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['pacmap']

        elif mnist_plot_dictionary['main'] == 'tsne':
            out_subplot = make_mnist_figure(df_mnist, tsne_base_version, index=clicked_index, is_subplot=True)
            
            mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'tsne'
            mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['pacmap']

        if mnist_embedding_dictionary['pacmap'] == 'subplot_1':
            mnist_plot_dictionary['main'] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = 'main'
        
            return out_main_plot, out_subplot, no_update, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['pacmap'] == 'subplot_2':
            mnist_plot_dictionary['main'] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = 'main'
        
            return out_main_plot, no_update, out_subplot, no_update, new_slider_div
        
        elif mnist_embedding_dictionary['pacmap'] == 'subplot_3':
            mnist_plot_dictionary['main'] = 'pacmap'
            mnist_embedding_dictionary['pacmap'] = 'main'
        
            return out_main_plot, no_update, no_update, out_subplot, new_slider_div


if __name__ == '__main__':
    app.run_server(debug=True)