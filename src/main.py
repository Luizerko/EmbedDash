import trimap
import umap
import pacmap
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
from models import train_and_predict, generate_latent_data, models


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


########################## FIGURES ##########################


### MNIST Data
fig = px.scatter(
    df_mnist, x='x', y='y', color='label',
    title="TRIMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x': False, 'y': False, 'image': False},
    width=800, height=640, size_max=10
).update_layout(fig_layout_dict)

umap_fig = px.scatter(
    df_mnist, x='x_umap', y='y_umap', color='label',
    title="UMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_umap': False, 'y_umap': False, 'image': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)

tsne_fig = px.scatter(
    df_mnist, x='x_tsne', y='y_tsne', color='label',
    title="T-SNE Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_tsne': False, 'y_tsne': False, 'image': False},
    width=400, height=320
).update_layout(small_fig_layout_dict)

pacmap_fig = px.scatter(
    df_mnist, x='x_pacmap', y='y_pacmap', color='label',
    title="PaCMAP Embedding",
    labels={'color': 'Digit', 'label': 'Label'},
    hover_data={'label': False, 'x_pacmap': False, 'y_pacmap': False, 'image': False},
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

pacmap_mammoth = px.scatter_3d(
    df_mammoth, x='x_pacmap', y='y_pacmap', z='z_pacmap', color='label',
    title="PaCMAP Embedding",
    hover_data={'label': False, 'x_pacmap': False, 'y_pacmap': False, 'z_pacmap': False},
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


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='MNIST Data', children=[
            html.Div([
                ### Upper row
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='scatter-plot', 
                            figure=fig,
                            style={}
                        ),
                        html.Div([
                            html.Button('PARAMETERS GO HERE', id='placeholdedr-button1', n_clicks=0)
                        ]),
                    ], style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'border-radius': '15px', 'margin': '10px', 'justify-content': 'space-around', 'align-items': 'center', 'background': '#FFFFFF', 'width': '95%'}),

                    ### Lower row
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='UMAP-plot', 
                                figure=umap_fig,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='T-SNE-plot',
                                figure=tsne_fig,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='PaCMAP-plot',
                                figure=pacmap_fig,
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
                        id='original-mammoth',
                        figure=original_mammoth,
                        style={}
                        )
                    ]),
                    html.Div([
                        dcc.Graph(
                            id='trimap-mammoth',
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
                                id='umap-mammoth',
                                figure=umap_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='t-sne-mammoth',
                                figure=tsne_mammoth,
                                style={"width": "100%", "display": "inline-block", 'height': '300px'}
                            ),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

                        html.Div([
                            dcc.Graph(
                                id='pacmap-mammoth',
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
            df_mnist['x'] = df_mnist['x'] + df_mnist['x_shift']
            df_mnist['y'] = df_mnist['y'] + df_mnist['y_shift']
        else:
            df_mnist['x'] = df_mnist['x'] - df_mnist['x_shift']
            df_mnist['y'] = df_mnist['y'] - df_mnist['y_shift']

        updated_fig = px.scatter(
            df_mnist, x='x', y='y', color='label',
            title="TRIMAP embeddings on MNIST",
            labels={'color': 'Digit', 'label': 'Label'},
            hover_data={'label': False, 'x': False, 'y': False, 'image': False},
            width=800, height=640, size_max=10
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