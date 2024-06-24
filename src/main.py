import pandas as pd
import os

import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State, callback

from layouts import fig_layout_dict, small_fig_layout_dict, fig_layout_dict_mammoth
from models import generate_latent_data


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
                    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
                    width=800, height=640, size_max=10
                ).update_layout(fig_layout_dict)

umap_mnist = px.scatter(
                    df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                    title="UMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)

tsne_mnist = px.scatter(
                    df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                    title="T-SNE Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)

pacmap_mnist = px.scatter(
                    df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                    title="PACMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                    width=400, height=320
                ).update_layout(small_fig_layout_dict)


### Mammoth Data
original_mammoth = px.scatter_3d(
    df_mammoth, x='x', y='y', z='z', color='label',
    title="Original Mammoth Data",
    hover_data={'label': False, 'x': False, 'y': False, 'z': False},
    width=600, height=480
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

trimap_mammoth = px.scatter_3d(
    df_mammoth, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', z='z_trimap_nin_12_nout_4', color='label',
    title="TRIMAP Embedding",
    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'z_trimap_nin_12_nout_4': False},
    width=600, height=480
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

umap_mammoth = px.scatter_3d(
    df_mammoth, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', z='z_umap_nneighbors_15_mindist_0.1', color='label',
    title="UMAP Embedding",
    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'z_umap_nneighbors_15_mindist_0.1': False},
    width=600, height=480
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

tsne_mammoth = px.scatter_3d(
    df_mammoth, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', z='z_tsne_perp_30_exa_12', color='label',
    title="T-SNE Embedding",
    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'z_tsne_perp_30_exa_12': False},
    width=600, height=480
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))

pacmap_mammoth = px.scatter_3d(
    df_mammoth, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', z='z_pacmap_nneighbors_10_init_pca', color='label',
    title="PACMAP Embedding",
    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'z_pacmap_nneighbors_10_init_pca': False},
    width=600, height=480
).update_layout(fig_layout_dict_mammoth).update_traces(marker=dict(size=1))


### Latent Data
trimap_latent = px.scatter(
    df_latent, x='x', y='y', color='label',
    title="TRIMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x': False, 'y': False},
    width=700, height=480, size_max=10
).update_layout(fig_layout_dict)

umap_latent = px.scatter(
    df_latent, x='x_umap', y='y_umap', color='label',
    title="UMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_umap': False, 'y_umap': False},
    width=700, height=480
).update_layout(fig_layout_dict)

tsne_latent = px.scatter(
    df_latent, x='x_tsne', y='y_tsne', color='label',
    title="T-SNE Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_tsne': False, 'y_tsne': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)


pacmap_fig_latent = px.scatter(
    df_latent, x='x_pacmap', y='y_pacmap', color='label',
    title="PaCMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_pacmap': False, 'y_pacmap': False, 'image': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)


####################### APP LAYOUT #######################

multiplicative_values = [2**i for i in range(1, 4)]
marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

app.layout = html.Div([
    dcc.Tabs([

        ### MNIST Data
        dcc.Tab(label='MNIST Data', children=[
            html.Div([
                ### Upper row
                html.Div([
                    html.Div([
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

                ], style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap', 'width': '85vw'}),

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

        ### Mammoth Data
        dcc.Tab(label='Mammoth Data', children=[
            html.Div([
                #Left grid
                html.Div([
                        html.Div([
                            dcc.Graph(
                                id='mammoth_trimap_plot',
                                figure=trimap_mammoth,
                                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}
                                )
                        ]),
                        html.Div([
                            dcc.Graph(
                                id='mammoth_umap_plot',
                                figure=umap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_tsne_plot',
                                figure=tsne_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_pacmap_plot',
                                figure=pacmap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ]),

                ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '65%', 'height': '100%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),
                
                #Right column
                html.Div([
                    html.Div([
                        dcc.Graph(
                        id='mammoth_original_plot',
                        figure=original_mammoth,
                        style={}
                        )
                    ]),
                ], style={'padding': '20px', 'display': 'flex', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '30%', 'justify-content': 'center', 'align-items': 'center', 'height': '40%'}),
                
            ], style={"display": "flex", "flexDirection": "column", 'flex-wrap': 'wrap', "padding": "20px", "background": "#E5F6FD", 'height': '90vh', 'justify-content': 'center'})
        ]),
        
        ### Latent Data
        dcc.Tab(label='CIFAR Data in Latent Space', children=[
            html.Div([
                # Upper row
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='scatter-plot-latent',
                            figure=fig_latent,
                            style={}
                        )
                    ]),
                ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '95%', 'justify-content': 'space-around', 'align-items': 'center'}),

                # Lower row
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='UMAP-plot-latent',
                            figure=umap_fig_latent,
                            style={}
                        )
                    ]),
                    html.Div([
                        dcc.Graph(
                            id='T-SNE-plot-latent',
                            figure=tsne_fig_latent,
                            style={"width": "100%", "display": "inline-block", 'height': '300px'}
                        ),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                    html.Div([
                        dcc.Graph(
                            id='PaCMAP-plot-latent',
                            figure=pacmap_fig_latent,
                            style={"width": "100%", "display": "inline-block", 'height': '300px'}
                        ),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '95%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),
                
            ], style={"display": "flex", "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'flex-wrap': 'wrap', 'height': '90vh'})
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
                hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
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
                hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
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
                hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
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
                hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                width=800, height=640, size_max=10
            )
            updated_fig.update_layout(fig_layout_dict)
            return updated_fig, n_clicks
    
    return current_figure, n_clicks


@app.callback(
    Output('mnist_main_plot', 'figure', allow_duplicate=True),
    [Input('mnist_slider_1', 'value'),
     Input('mnist_slider_2', 'value')],
    prevent_initial_call=True
)
def sliders(slider_value_1, slider_value_2):
    if mnist_plot_dictionary["main"] == "trimap":
        trimap_nout_values=[2, 4, 8]
        updated_fig = px.scatter(
            df_mnist, x='x_trimap_nin_'+str(slider_value_1)+'_nout_'+str(trimap_nout_values[slider_value_2]), y='y_trimap_nin_'+str(slider_value_1)+'_nout_'+str(trimap_nout_values[slider_value_2]), color='label',
            title="TRIMAP Embedding ",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': False, 'x_trimap_nin_'+str(slider_value_1)+'_nout_'+str(trimap_nout_values[slider_value_2]): False, 'y_trimap_nin_'+str(slider_value_1)+'_nout_'+str(trimap_nout_values[slider_value_2]): False, 'image': False},
            width=800, height=640, size_max=10
        )
        
    elif mnist_plot_dictionary["main"] == "umap":
        umap_nneighbors_values=[5, 15, 45]
        umap_mindist_values=[0.0, 0.1, 0.5]
        updated_fig = px.scatter(
            df_mnist, x='x_umap_nneighbors_'+str(umap_nneighbors_values[slider_value_1])+'_mindist_'+str(umap_mindist_values[slider_value_2]), y='y_umap_nneighbors_'+str(umap_nneighbors_values[slider_value_1])+'_mindist_'+str(umap_mindist_values[slider_value_2]), color='label',
            title="UMAP Embedding ",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': False, 'x_umap_nneighbors_'+str(umap_nneighbors_values[slider_value_1])+'_mindist_'+str(umap_mindist_values[slider_value_2]): False, 'y_umap_nneighbors_'+str(umap_nneighbors_values[slider_value_1])+'_mindist_'+str(umap_mindist_values[slider_value_2]): False, 'image': False},
            width=800, height=640, size_max=10
        )
        
    elif mnist_plot_dictionary["main"] == "tsne":
        tsne_exa_values=[6, 12, 24]
        updated_fig = px.scatter(
            df_mnist, x='x_tsne_perp_'+str(slider_value_1)+'_exa_'+str(tsne_exa_values[slider_value_2]), y='y_tsne_perp_'+str(slider_value_1)+'_exa_'+str(tsne_exa_values[slider_value_2]), color='label',
            title="TSNE Embedding ",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': False, 'x_tsne_perp_'+str(slider_value_1)+'_exa_'+str(tsne_exa_values[slider_value_2]): False, 'y_tsne_perp_'+str(slider_value_1)+'_exa_'+str(tsne_exa_values[slider_value_2]): False, 'image': False},
            width=800, height=640, size_max=10
        )

    elif mnist_plot_dictionary["main"] == "pacmap":
        pacmap_nneighbors_values=[5, 10, 20]
        pacmap_init_values=['pca', 'random']
        updated_fig = px.scatter(
            df_mnist, x='x_pacmap_nneighbors_'+str(pacmap_nneighbors_values[slider_value_1])+'_init_'+pacmap_init_values[slider_value_2], y='y_pacmap_nneighbors_'+str(pacmap_nneighbors_values[slider_value_1])+'_init_'+pacmap_init_values[slider_value_2], color='label',
            title="PACMAP Embedding ",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': False, 'x_pacmap_nneighbors_'+str(pacmap_nneighbors_values[slider_value_1])+'_init_'+pacmap_init_values[slider_value_2]: False, 'y_pacmap_nneighbors_'+str(pacmap_nneighbors_values[slider_value_1])+'_init_'+pacmap_init_values[slider_value_2]: False, 'image': False},
            width=800, height=640, size_max=10
        )
    
    updated_fig.update_layout(fig_layout_dict)
    return updated_fig

#     updated_fig.update_layout(fig_layout_dict)

#     return updated_fig

# @callback(
#     [Output('hover-image', 'src'),
#      Output('hover-index', 'children'),
#      Output('scatter-plot', 'hoverData'),
#      Output('UMAP-plot', 'hoverData'),
#      Output('T-SNE-plot', 'hoverData')],
#     [Input('scatter-plot', 'hoverData'),
#      Input('UMAP-plot', 'hoverData'),
#      Input('T-SNE-plot', 'hoverData')]
# )
# def display_hover_image(MainhoverData, UMAPhoverData, TSNEhoverData):
    
#     # if you are hovering over any of the input images, get that hoverData
#     hoverData = None
#     inputs = [MainhoverData, UMAPhoverData, TSNEhoverData]
#     for inp in inputs:
#         if inp is not None:
#             hoverData = inp
#             break
    
#     if hoverData is None:
#         return '', '', None, None, None

#     original_label = hoverData['points'][0]['customdata'][0]
#     original_image = hoverData['points'][0]['customdata'][1]

#     return original_image, f'Label: {original_label}', None, None, None


# @callback(
#     [Output('click-image', 'src'),
#      Output('click-index', 'children'),
#      Output('mnist_trimap_plot', 'clickData'),
#      Output('mnist_umap_plot', 'clickData'),
#      Output('mnist_tsne_plot', 'clickData')],
#     [Input('mnist_trimap_plot', 'clickData'),
#      Input('mnist_umap_plot', 'clickData'),
#      Input('mnist_tsne_plot', 'clickData')]
# )
# def display_click_image(MainclickData, UMAPclickData, TSNEclickData):
#     clickData = None
#     inputs = [MainclickData, UMAPclickData, TSNEclickData]
#     for inp in inputs:
#         if inp is not None:
#             clickData = inp
#             break

#     if clickData is not None:
#         original_label = clickData['points'][0]['customdata'][0]
#         original_image = clickData['points'][0]['customdata'][1]

#         return original_image, f'Label: {original_label}', None, None, None
#     return '', '', None, None, None


@app.callback(
    [#Output('mnist_dd_choose_embedding', 'dropdown_value'),
     Output('mnist_main_plot', 'figure'),
     Output('mnist_subplot_1', 'figure'),
     Output('mnist_subplot_2', 'figure'),
     Output('mnist_subplot_3', 'figure'), 
     Output('sliders_div', 'children')],
    [Input('mnist_dd_choose_embedding', 'value')],
    [State('mnist_main_plot', 'figure'),
     State('mnist_subplot_1', 'figure'),
     State('mnist_subplot_2', 'figure'),
     State('mnist_subplot_3', 'figure'),
     State('sliders_div', 'children')]
)
def switch_main_img(dropdown_value, in_main_plot, in_subplot_1, in_subplot_2, in_subplot_3, old_slider_div):
    if dropdown_value != mnist_plot_dictionary['main']:
        if dropdown_value == 'trimap':
            out_main_plot = px.scatter(
                df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                title="TRIMAP Embedding ",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
                width=800, height=640, size_max=10
            )
            out_main_plot.update_layout(fig_layout_dict)

            multiplicative_values = [2**i for i in range(1, 4)]
            marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

            new_slider_div = [
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
            ]

            if mnist_plot_dictionary['main'] == 'umap':
                out_subplot = px.scatter(
                    df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                    title="UMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'umap'
                mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['trimap']
            
            elif mnist_plot_dictionary['main'] == 'tsne':
                out_subplot = px.scatter(
                    df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                    title="TSNE Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)

                mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'tsne'
                mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['trimap']

            elif mnist_plot_dictionary['main'] == 'pacmap':
                out_subplot = px.scatter(
                    df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                    title="PACMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['trimap']] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['trimap']

            if mnist_embedding_dictionary['trimap'] == 'subplot_1':
                mnist_plot_dictionary['main'] = 'trimap'
                mnist_embedding_dictionary['trimap'] = 'main'
            
                return out_main_plot, out_subplot, in_subplot_2, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['trimap'] == 'subplot_2':
                mnist_plot_dictionary['main'] = 'trimap'
                mnist_embedding_dictionary['trimap'] = 'main'
            
                return out_main_plot, in_subplot_1, out_subplot, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['trimap'] == 'subplot_3':
                mnist_plot_dictionary['main'] = 'trimap'
                mnist_embedding_dictionary['trimap'] = 'main'
            
                return out_main_plot, in_subplot_1, in_subplot_2, out_subplot, new_slider_div
            
        elif dropdown_value == 'umap':
            out_main_plot = px.scatter(
                df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                title="UMAP Embedding",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
                width=800, height=640, size_max=10
            )
            out_main_plot.update_layout(fig_layout_dict)

            multiplicative_values_1 = [5, 15, 45]
            marks_1 = {i: str(value) for i, value in enumerate(multiplicative_values_1)}

            multiplicative_values_2 = [0.0, 0.1, 0.5]
            marks_2 = {i: str(value) for i, value in enumerate(multiplicative_values_2)}

            new_slider_div = [
                html.Label('Number of Neighbors:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=0, max=len(multiplicative_values_1)-1, value=1, step=None, marks=marks_1,
                id='mnist_slider_1',
                ),
                html.Label('Minimum Distance:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=0, max=len(multiplicative_values_2)-1, value=1, step=None, marks=marks_2,
                id='mnist_slider_2',
                ),
                html.Div([], style={'height': '50px'}),
                html.Label('Choose the Embedding:', style={}), 
                dcc.Dropdown(
                    options=['trimap', 'umap', 'tsne', 'pacmap'], value='umap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
                ),
            ]

            if mnist_plot_dictionary['main'] == 'trimap':
                out_subplot = px.scatter(
                    df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                    title="TRIMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'trimap'
                mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['umap']
            
            elif mnist_plot_dictionary['main'] == 'tsne':
                out_subplot = px.scatter(
                    df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                    title="TSNE Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)

                mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'tsne'
                mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['umap']

            elif mnist_plot_dictionary['main'] == 'pacmap':
                out_subplot = px.scatter(
                    df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                    title="PACMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['umap']] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['umap']

            if mnist_embedding_dictionary['umap'] == 'subplot_1':
                mnist_plot_dictionary['main'] = 'umap'
                mnist_embedding_dictionary['umap'] = 'main'
            
                return out_main_plot, out_subplot, in_subplot_2, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['umap'] == 'subplot_2':
                mnist_plot_dictionary['main'] = 'umap'
                mnist_embedding_dictionary['umap'] = 'main'
            
                return out_main_plot, in_subplot_1, out_subplot, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['umap'] == 'subplot_3':
                mnist_plot_dictionary['main'] = 'umap'
                mnist_embedding_dictionary['umap'] = 'main'
            
                return out_main_plot, in_subplot_1, in_subplot_2, out_subplot, new_slider_div
            
        elif dropdown_value == 'tsne':
            out_main_plot = px.scatter(
                df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                title="TSNE Embedding",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
                width=800, height=640, size_max=10
            )
            out_main_plot.update_layout(fig_layout_dict)

            # perplexity 30 min=15 max=45
            # early_exageration 12 min=6 max=24
            multiplicative_values = [6, 12, 24]
            marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

            new_slider_div = [
                html.Label('Perplexity:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=15, max=45, value=30, step=15,
                id='mnist_slider_1',
                ),
                html.Label('Early Exagerration:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=0, max=len(multiplicative_values)-1, value=1, step=None, marks=marks,
                id='mnist_slider_2',
                ),
                html.Div([], style={'height': '50px'}),
                html.Label('Choose the Embedding:', style={}), 
                dcc.Dropdown(
                    options=['trimap', 'umap', 'tsne', 'pacmap'], value='tsne', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
                ),
            ]

            if mnist_plot_dictionary['main'] == 'trimap':
                out_subplot = px.scatter(
                    df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                    title="TRIMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'trimap'
                mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['tsne']
            
            elif mnist_plot_dictionary['main'] == 'umap':
                out_subplot = px.scatter(
                    df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                    title="UMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)

                mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'umap'
                mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['tsne']

            elif mnist_plot_dictionary['main'] == 'pacmap':
                out_subplot = px.scatter(
                    df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                    title="PACMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['tsne']] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = mnist_embedding_dictionary['tsne']

            if mnist_embedding_dictionary['tsne'] == 'subplot_1':
                mnist_plot_dictionary['main'] = 'tsne'
                mnist_embedding_dictionary['tsne'] = 'main'
            
                return out_main_plot, out_subplot, in_subplot_2, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['tsne'] == 'subplot_2':
                mnist_plot_dictionary['main'] = 'tsne'
                mnist_embedding_dictionary['tsne'] = 'main'
            
                return out_main_plot, in_subplot_1, out_subplot, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['tsne'] == 'subplot_3':
                mnist_plot_dictionary['main'] = 'tsne'
                mnist_embedding_dictionary['tsne'] = 'main'
            
                return out_main_plot, in_subplot_1, in_subplot_2, out_subplot, new_slider_div

        elif dropdown_value == 'pacmap':
            out_main_plot = px.scatter(
                df_mnist, x='x_pacmap_nneighbors_10_init_pca', y='y_pacmap_nneighbors_10_init_pca', color='label',
                title="PACMAP Embedding",
                labels={'color': 'Digit', 'label': 'Label'},
                hover_data={'label': False, 'x_pacmap_nneighbors_10_init_pca': False, 'y_pacmap_nneighbors_10_init_pca': False, 'image': False},
                width=800, height=640, size_max=10
            )
            out_main_plot.update_layout(fig_layout_dict)

            multiplicative_values_1 = [5, 10, 20]
            marks_1 = {i: str(value) for i, value in enumerate(multiplicative_values_1)}

            multiplicative_values_2 = ['pca', 'random']
            marks_2 = {i: str(value) for i, value in enumerate(multiplicative_values_2)}

            new_slider_div = [
                html.Label('Number of Neighbors:', id='title_slider_1', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=0, max=len(multiplicative_values_1)-1, value=1, step=None, marks=marks_1,
                id='mnist_slider_1',
                ),
                html.Label('Initialization:', id='title_slider_2', style={'margin-top': '20px', 'margin-bottom': '10px'}),
                dcc.Slider(min=0, max=len(multiplicative_values_2)-1, value=0, step=None, marks=marks_2,
                id='mnist_slider_2',
                ),
                html.Div([], style={'height': '50px'}),
                html.Label('Choose the Embedding:', style={}), 
                dcc.Dropdown(
                    options=['trimap', 'umap', 'tsne', 'pacmap'], value='pacmap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
                ),
            ]

            if mnist_plot_dictionary['main'] == 'trimap':
                out_subplot = px.scatter(
                    df_mnist, x='x_trimap_nin_12_nout_4', y='y_trimap_nin_12_nout_4', color='label',
                    title="TRIMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_trimap_nin_12_nout_4': False, 'y_trimap_nin_12_nout_4': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'trimap'
                mnist_embedding_dictionary['trimap'] = mnist_embedding_dictionary['pacmap']
            
            elif mnist_plot_dictionary['main'] == 'umap':
                out_subplot = px.scatter(
                    df_mnist, x='x_umap_nneighbors_15_mindist_0.1', y='y_umap_nneighbors_15_mindist_0.1', color='label',
                    title="UMAP Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_umap_nneighbors_15_mindist_0.1': False, 'y_umap_nneighbors_15_mindist_0.1': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)

                mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'umap'
                mnist_embedding_dictionary['umap'] = mnist_embedding_dictionary['pacmap']

            elif mnist_plot_dictionary['main'] == 'tsne':
                out_subplot = px.scatter(
                    df_mnist, x='x_tsne_perp_30_exa_12', y='y_tsne_perp_30_exa_12', color='label',
                    title="TSNE Embedding",
                    labels={'color': 'Digit', 'label': 'Label'},
                    hover_data={'label': False, 'x_tsne_perp_30_exa_12': False, 'y_tsne_perp_30_exa_12': False, 'image': False},
                    width=400, height=320, size_max=10
                )
                out_subplot.update_layout(small_fig_layout_dict)
                
                mnist_plot_dictionary[mnist_embedding_dictionary['pacmap']] = 'tsne'
                mnist_embedding_dictionary['tsne'] = mnist_embedding_dictionary['pacmap']

            if mnist_embedding_dictionary['pacmap'] == 'subplot_1':
                mnist_plot_dictionary['main'] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = 'main'
            
                return out_main_plot, out_subplot, in_subplot_2, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['pacmap'] == 'subplot_2':
                mnist_plot_dictionary['main'] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = 'main'
            
                return out_main_plot, in_subplot_1, out_subplot, in_subplot_3, new_slider_div
            
            elif mnist_embedding_dictionary['pacmap'] == 'subplot_3':
                mnist_plot_dictionary['main'] = 'pacmap'
                mnist_embedding_dictionary['pacmap'] = 'main'
            
                return out_main_plot, in_subplot_1, in_subplot_2, out_subplot, new_slider_div

    #return dropdown_value, in_main_plot, in_subplot_1, in_subplot_2, in_subplot_3
    return in_main_plot, in_subplot_1, in_subplot_2, in_subplot_3, old_slider_div

if __name__ == '__main__':
    app.run_server(debug=True)