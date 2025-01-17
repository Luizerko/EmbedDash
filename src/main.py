import pandas as pd
import os

from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from models import generate_latent_data

from utils import make_mnist_figure, make_mammoth_figure, make_latent_figure, get_button_name


####################### VARIABLES #######################

# Paths to cache files
data_path = Path("data/")
if not os.path.exists(data_path):
    os.makedirs(data_path)
dataframe_mnist_path = data_path / "final_mnist_param_grid_data.pkl"
dataframe_mammoth_path = data_path / "mammoth_param_grid_data.pkl"
latent_data_path = data_path / "latent_data.pkl"

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="EmbedDash", update_title=None)


####################### DATA PREPROCESSING #######################


### MNIST Data
if dataframe_mnist_path.exists():
    # Load the DataFrame from the file
    df_mnist = pd.read_pickle(dataframe_mnist_path)
    df_mnist['label'] = df_mnist['label'].astype('category')
    df_mnist['label'] = df_mnist['label'].cat.set_categories(sorted(df_mnist['label'].unique()))


### Mammoth Data
if dataframe_mammoth_path.exists():
    # Load the DataFrame from the file
    df_mammoth = pd.read_pickle(dataframe_mammoth_path)

    df_mammoth['index'] = range(0, len(df_mammoth))


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
trimap_mnist = make_mnist_figure(df_mnist, 'trimap_nin_12_nout_4')

umap_mnist = make_mnist_figure(df_mnist, 'umap_nneighbors_15_mindist_0.1', is_subplot=True)

tsne_mnist = make_mnist_figure(df_mnist, 'tsne_perp_30_exa_12', is_subplot=True)

pacmap_mnist = make_mnist_figure(df_mnist, 'pacmap_nneighbors_10_init_pca', is_subplot=True)


### Mammoth Data
original_mammoth = make_mammoth_figure(df_mammoth, 'original')

trimap_mammoth = make_mammoth_figure(df_mammoth, 'trimap_nin_12_nout_4')

umap_mammoth = make_mammoth_figure(df_mammoth, 'umap_nneighbors_15_mindist_0.1')

tsne_mammoth = make_mammoth_figure(df_mammoth, 'tsne_perp_30_exa_12')

pacmap_mammoth = make_mammoth_figure(df_mammoth, 'pacmap_nneighbors_10_init_pca')



### Latent Data
trimap_latent = make_latent_figure(df_latent, 'trimap')

umap_latent = make_latent_figure(df_latent, 'umap')

tsne_latent = make_latent_figure(df_latent, 'tsne')

pacmap_latent = make_latent_figure(df_latent, 'pacmap')


####################### APP LAYOUT #######################

multiplicative_values = [2**i for i in range(1, 4)]
marks = {i: str(value) for i, value in enumerate(multiplicative_values)}

information_mnist = """
This button shifts the centroids of clusters on the embedding space so that they respect the euclidian distances of the centroids on the latent space. A second click reverses this operation.
"""
information_latent = """
By using the outputs from the second-to-last layer of EfficientNetV2L, a classification neural network trained on the CIFAR dataset, projecting the data onto a high-dimensional latent space before computing the embeddings to simulate a more realistic setup.
"""

app.layout = html.Div([
    dcc.Store(id='on-refresh'),
    dcc.Tabs([

        ### MNIST Data
        dcc.Tab(label='MNIST Data', children=[
            html.Div([
                html.Label('Assess Quality of Embeddings', id='title_mnist', style={'margin-top': '10px', 'margin-bottom': '10px', 'font-size': '50px', 'font-weight': 'bold', 'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
                html.Div([
                    ### Upper row
                    html.Div([
                        dcc.Store(id='version_parameters', storage_type='memory', data=['trimap_nin_12_nout_4', 'umap_nneighbors_15_mindist_0.1', 'tsne_perp_30_exa_12', 'pacmap_nneighbors_10_init_pca']),
                        dcc.Store(id='versions_on_latent', storage_type='memory', data=[False, False, False, False]),
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
                            html.Div([
                                html.Button('See Data Distribution on Latent Space', id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '20px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                                dbc.Button(
                                    html.I(className="bi bi-info-circle-fill me-2"),
                                    id="info_button_mnist", 
                                    color="info", 
                                    className="me-2", 
                                    style={"fontSize": "1.5rem", "background-color": "white", "boder-radius": "10px", "border-width": "0px"}
                                ),
                                dbc.Popover(
                                    [
                                        dbc.PopoverBody(dcc.Markdown(information_mnist, style={"font-size": "18px", 'color': '#FFFFFF'})),
                                    ],
                                    target="info_button_mnist",
                                    trigger="click",
                                    style = {'background-color': '#595959'}
                                ),
        
                            ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),
                            html.Div([], style={'height': '50px'}),
                            html.Label('Choose Embedding Technique:', style={}), 
                            dcc.Dropdown(
                                options=['trimap', 'umap', 'tsne', 'pacmap'], value='trimap', placeholder='Choose a different plot', id='mnist_dd_choose_embedding', style={'width': '300px', 'margin-top': '10px'}
                            ),
                        ], id='sliders_div', style={'width': '500px', 'height': '500px'}),
                    ], style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'border-radius': '15px', 'margin': '10px', 'justify-content': 'space-around', 'align-items': 'center', 'background': '#FFFFFF', 'width': '100%'}),

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

                    ], style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'border-radius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '100%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),

                ], style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap', 'width': '80vw', 'justify-content': 'center'}),

                ### Right column
                html.Div([

                    # Box for images
                    html.Div([
                        html.H3("Hover Sample", style={'text-align': 'center', 'font-weight': 'bold', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            html.Img(id='hover-image-mnist', style={'height': '200px'}),
                            html.Div(id='hover-index-mnist', style={'font-family': 'Arial', 'padding': '10px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                        
                        html.H3("Click Sample", style={'text-align': 'center', 'font-family': 'Arial', 'font-weight': 'bold', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            dcc.Store(id='clicked-index-mnist', storage_type='memory', data=None),
                            html.Img(id='click-image-mnist', style={'height': '200px'}),
                            html.Div(id='click-index-mnist', style={'font-family': 'Arial', 'padding': '10px'})
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
                    ], style={'padding': '20px', 'border-radius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '85%'}),

                ], style={'display': 'flex', 'flex-direction': 'column', 'border-radius': '15px', 'margin': '10px', 'align-items': 'center', 'width': '15vw', 'justify-content': 'center'}),

            ], style={"display": "flex", 'flex-wrap': 'wrap', "flexDirection": "row", "padding": "20px", "background": "#E5F6FD", 'height': '94vh', 'justify-content': 'space-around'})
        ]),

        ### Mammoth Data
        dcc.Tab(label='Mammoth Data', children=[
            html.Div([
                html.Label('Global Structure Preservation', id='title_mammoth', style={'margin-top': '10px', 'margin-bottom': '10px', 'font-size': '50px', 'font-weight': 'bold', 'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
                html.Div([
                    #Left grid
                    html.Div([
                        dcc.Store(id='clicked-index-mammoth', storage_type='memory', data=None),
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
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_tsne_plot',
                                figure=tsne_mammoth,
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='mammoth_pacmap_plot',
                                figure=pacmap_mammoth,
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),
                    ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '65%', 'height': '90%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),
                    #Right column
                    html.Div([
                        html.Div([
                            dcc.Graph(
                            id='mammoth_original_plot',
                            figure=original_mammoth,
                            style={}
                            )
                        ]),
                    ], style={'padding': '20px', 'display': 'flex', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '30%', 'justify-content': 'center', 'align-items': 'center', 'height': '50%'}),
                    
                ], style={"display": "flex", "flexDirection": "column", 'flex-wrap': 'wrap', 'justify-content': 'center', 'height': '90%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap', "background": "#E5F6FD", 'height': '94vh', 'padding': '20px'}),
        ]),
        
        ### Latent Data
        dcc.Tab(label='CIFAR Data', children=[
            html.Div([
                html.Div([
                    html.Label('Embeddings of Latents from a NN', id='title_latent', style={'font-size': '50px', 'font-weight': 'bold'}),
                    dbc.Button(
                        html.I(className="bi bi-info-circle-fill me-2"),
                        id="info_button_latent", 
                        color="info", 
                        className="me-2", 
                        style={"fontSize": "1.5rem", "background-color": "#E5F6FD", "boder-radius": "10px", "border-width": "0px"}
                    ),
                    dbc.Popover(
                        [
                            dbc.PopoverBody(dcc.Markdown(information_latent, style={"font-size": "18px", 'color': '#FFFFFF'})),
                        ],
                        target="info_button_latent",
                        trigger="click",
                        placement="right",
                        style = {'background-color': '#595959'}
                    )
                ], style={'margin-top': '10px', 'margin-bottom': '10px', 'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
                html.Div([
                    ### Left grid
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='latent_trimap_plot',
                                figure=trimap_latent,
                                style={"width": "100%", "display": "inline-block"}
                                )
                        ]),
                        html.Div([
                            dcc.Graph(
                                id='latent_umap_plot',
                                figure=umap_latent,
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='latent_tsne_plot',
                                figure=tsne_latent,
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='latent_pacmap_plot',
                                figure=pacmap_latent,
                                style={"width": "100%", "display": "inline-block"}
                            ),
                        ]),

                    ], style={'padding': '20px', 'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap', 'borderRadius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '100%', 'height': '100%', 'justify-content': 'space-around', 'align-items': 'flex-start'}),

                ], style={"display": "flex", "flexDirection": "column", 'flex-wrap': 'wrap', "padding": "20px", "background": "#E5F6FD", 'height': '90%', 'align-items': 'center', 'width': '80%'}),

                ### Right column
                html.Div([

                    # Box for images
                    html.Div([
                        dcc.Store(id='clicked-index-latent', storage_type='memory', data=None),
                        html.H3("Hover Sample", style={'text-align': 'center', 'font-weight': 'bold', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            html.Img(id='hover-image-latent', style={'height': '200px'}),
                            html.Div(id='hover-index-latent', style={'font-family': 'Arial', 'padding': '10px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                        
                        html.H3("Click Sample", style={'text-align': 'center', 'font-weight': 'bold', 'font-family': 'Arial', 'margin-top': '5px', 'margin-bottom': '5px'}),
                        
                        html.Div([
                            html.Img(id='click-image-latent', style={'height': '200px'}),
                            html.Div(id='click-index-latent', style={'font-family': 'Arial', 'padding': '10px'})
                        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
                    ], style={'padding': '20px', 'border-radius': '15px', 'background': '#FFFFFF', 'margin': '10px', 'width': '85%'}),

                ], style={'display': 'flex', 'flex-direction': 'column', 'border-radius': '15px', 'margin': '10px', 'align-items': 'center', 'width': '15%', 'justify-content': 'center'}),
            ], style={"display": "flex", "flexDirection": "row", 'flex-wrap': 'wrap', "padding": "20px", "background": "#E5F6FD", 'height': '94vh', 'justify-content': 'space-around'})
        ]),
    ])
])


####################### CALLBACKS #######################

# reload MNIST on page refresh
# necessary because the latent space distances function changes df_mnist directly and on page refresh this creates bugs
# we decided to do it that way so we didn't have to compute the updates for each point individually
@app.callback(
    Output('on-refresh', 'id'), 
    [Input('on-refresh', 'id')])
def reload_mnist(id):
    global df_mnist
    if dataframe_mnist_path.exists():
        # Load the DataFrame from the file
        df_mnist = pd.read_pickle(dataframe_mnist_path)
        df_mnist['label'] = df_mnist['label'].astype('category')
        df_mnist['label'] = df_mnist['label'].cat.set_categories(sorted(df_mnist['label'].unique()))
    raise PreventUpdate


# Callback for the latent space distances function
@app.callback(
    [Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('translate-button', 'n_clicks'),
     Output('translate-button', 'children'),
     Output('versions_on_latent', 'data')],
    [Input('translate-button', 'n_clicks')],
    [State('mnist_dd_choose_embedding', 'value'),
     State('version_parameters', 'data'),
     State('clicked-index-mnist', 'data'),
     State('versions_on_latent', 'data')],
     prevent_initial_call=True
)
def update_plot(n_clicks, dropdown_value, version_states, clicked_index, versions_on_latent):

    # stop from triggering when button is created
    if n_clicks == 0:
        raise PreventUpdate

    # get the versions to update
    if dropdown_value == 'trimap':
        type_index = 0 # type_index to not have to re-use if-else the entire time
        versions = ['trimap_nin_12_nout_4', 'trimap_nin_8_nout_4', 'trimap_nin_16_nout_4',
        'trimap_nin_12_nout_2', 'trimap_nin_8_nout_2', 'trimap_nin_16_nout_2',
        'trimap_nin_12_nout_8', 'trimap_nin_8_nout_8', 'trimap_nin_16_nout_8']
    elif dropdown_value == 'umap':
        type_index = 1
        versions = ['umap_nneighbors_5_mindist_0.1', 'umap_nneighbors_15_mindist_0.1', 'umap_nneighbors_45_mindist_0.1', 
        'umap_nneighbors_5_mindist_0.0', 'umap_nneighbors_15_mindist_0.0', 'umap_nneighbors_45_mindist_0.0',
        'umap_nneighbors_5_mindist_0.5', 'umap_nneighbors_15_mindist_0.5', 'umap_nneighbors_45_mindist_0.5']
    elif dropdown_value == 'tsne':
        type_index = 2
        versions = ['tsne_perp_30_exa_12', 'tsne_perp_15_exa_12', 'tsne_perp_45_exa_12', 
        'tsne_perp_30_exa_6', 'tsne_perp_15_exa_6', 'tsne_perp_45_exa_6', 
        'tsne_perp_30_exa_24', 'tsne_perp_15_exa_24', 'tsne_perp_45_exa_24', ]
    elif dropdown_value == 'pacmap':
        type_index = 3
        versions = ['pacmap_nneighbors_10_init_pca', 'pacmap_nneighbors_5_init_pca',  'pacmap_nneighbors_20_init_pca', 
        'pacmap_nneighbors_10_init_random', 'pacmap_nneighbors_5_init_random',  'pacmap_nneighbors_20_init_random']

    # if it is on latent, return to embedding
    if versions_on_latent[type_index]:
        versions_on_latent[type_index] = False
        button_title = get_button_name(False)
        for version in versions:
            df_mnist['x_'+version] = df_mnist['x_'+version] - df_mnist['x_shift_'+version]
            df_mnist['y_'+version] = df_mnist['y_'+version] - df_mnist['y_shift_'+version]
    # if it is on embedding, set to latent
    else:
        versions_on_latent[type_index] = True
        button_title = get_button_name(True)
        for version in versions:
            df_mnist['x_'+version] = df_mnist['x_'+version] + df_mnist['x_shift_'+version]
            df_mnist['y_'+version] = df_mnist['y_'+version] + df_mnist['y_shift_'+version]

    version = version_states[type_index]
    updated_fig = make_mnist_figure(df_mnist, version, index=clicked_index, is_subplot=False)

    return updated_fig, n_clicks, button_title, versions_on_latent


@app.callback(
    [Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('version_parameters', 'data', allow_duplicate=True)],
    [Input('mnist_slider_1', 'value'),
     Input('mnist_slider_2', 'value')],
    [State('version_parameters', 'data'),
     State('clicked-index-mnist', 'data')],
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


@callback(
    [Output('hover-image-mnist', 'src'),
     Output('hover-index-mnist', 'children'),
     Output('mnist_main_plot', 'hoverData'),
     Output('mnist_subplot_1', 'hoverData'),
     Output('mnist_subplot_2', 'hoverData'),
     Output('mnist_subplot_3', 'hoverData')],
    [Input('mnist_main_plot', 'hoverData'),
     Input('mnist_subplot_1', 'hoverData'),
     Input('mnist_subplot_2', 'hoverData'),
     Input('mnist_subplot_3', 'hoverData')]

)
def display_hover_image_mnist(MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData):
    
    # if you are hovering over any of the input images, get that hoverData
    hoverData = None
    inputs = [MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData]
    for inp in inputs:
        if inp is not None:
            hoverData = inp
            break
    
    if hoverData is None:
        return '', '', None, None, None, None

    original_index = hoverData['points'][0]['customdata'][0]

    df_row = df_mnist.loc[original_index]
    original_label = df_row['label']
    original_image = df_row['image']

    return original_image, f'Label: {original_label}', None, None, None, None


@callback(
    [Output('hover-image-latent', 'src'),
     Output('hover-index-latent', 'children'),
     Output('latent_trimap_plot', 'hoverData'),
     Output('latent_umap_plot', 'hoverData'),
     Output('latent_tsne_plot', 'hoverData'),
     Output('latent_pacmap_plot', 'hoverData')],
    [Input('latent_trimap_plot', 'hoverData'),
     Input('latent_umap_plot', 'hoverData'),
     Input('latent_tsne_plot', 'hoverData'),
     Input('latent_pacmap_plot', 'hoverData')]
)
def display_hover_image_latent(MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData):
    
    # if you are hovering over any of the input images, get that hoverData
    hoverData = None
    inputs = [MainhoverData, Sub1hoverData, Sub2hoverData, Sub3hoverData]
    for inp in inputs:
        if inp is not None:
            hoverData = inp
            break
    

    if hoverData is None:
        return '', '', None, None, None, None

    original_index = hoverData['points'][0]['customdata'][0]

    df_row = df_latent.loc[original_index]
    original_label = df_row['label']
    original_image = df_row['image']

    return original_image, f'Label: {original_label}', None, None, None, None



@callback(
    [Output('click-image-mnist', 'src'),
     Output('click-index-mnist', 'children'),
     Output('clicked-index-mnist', 'data'),
     Output('mnist_main_plot', 'figure', allow_duplicate=True),
     Output('mnist_subplot_1', 'figure', allow_duplicate=True),
     Output('mnist_subplot_2', 'figure', allow_duplicate=True),
     Output('mnist_subplot_3', 'figure', allow_duplicate=True),
     Output('mnist_main_plot', 'clickData'),
     Output('mnist_subplot_1', 'clickData'),
     Output('mnist_subplot_2', 'clickData'),
     Output('mnist_subplot_3', 'clickData')],
    [Input('mnist_main_plot', 'clickData'),
     Input('mnist_subplot_1', 'clickData'),
     Input('mnist_subplot_2', 'clickData'),
     Input('mnist_subplot_3', 'clickData')],
    [State('version_parameters', 'data')],
    prevent_initial_call=True
)
def display_click_image_mnist(MainclickData, Sub1clickData, Sub2clickData, Sub3clickData, version_parameters):
    clickData = None
    inputs = [MainclickData, Sub1clickData, Sub2clickData, Sub3clickData]
    for inp in inputs:
        if inp is not None:
            clickData = inp
            break

    if clickData is None:
        raise PreventUpdate

    original_index = clickData['points'][0]['customdata'][0]
    df_row = df_mnist.loc[original_index]
    original_label = df_row['label']
    original_image = df_row['image']

    trimap_base_version = version_parameters[0]
    umap_base_version = version_parameters[1]
    tsne_base_version = version_parameters[2]
    pacmap_base_version = version_parameters[3]

    for location, embedding in mnist_plot_dictionary.items():
        if location == 'main':
            if embedding == 'trimap':
                main_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=False)
            elif embedding == 'umap':
                main_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=False)
            elif embedding == 'tsne':
                main_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=False)
            elif embedding == 'pacmap':
                main_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=False)

        elif location == 'subplot_1':
            if embedding == 'trimap':
                sub1_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'umap':
                sub1_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'tsne':
                sub1_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
            elif embedding == 'pacmap':
                sub1_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
        
        elif location == 'subplot_2':
            if embedding == 'trimap':
                sub2_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'umap':
                sub2_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'tsne':
                sub2_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
            elif embedding == 'pacmap':
                sub2_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
        
        elif location == 'subplot_3':
            if embedding == 'trimap':
                sub3_img = make_mnist_figure(df_mnist, trimap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'umap':
                sub3_img = make_mnist_figure(df_mnist, umap_base_version, index=original_index, is_subplot=True)
            elif embedding == 'tsne':
                sub3_img = make_mnist_figure(df_mnist, tsne_base_version, index=original_index, is_subplot=True)
            elif embedding == 'pacmap':
                sub3_img = make_mnist_figure(df_mnist, pacmap_base_version, index=original_index, is_subplot=True)
        
    return original_image, f'Label: {original_label}', original_index, main_img, sub1_img, sub2_img, sub3_img, None, None, None, None


@callback(
    [Output('click-image-latent', 'src'),
     Output('click-index-latent', 'children'),
     Output('clicked-index-latent', 'data'),
     Output('latent_trimap_plot', 'figure', allow_duplicate=True),
     Output('latent_umap_plot', 'figure', allow_duplicate=True),
     Output('latent_tsne_plot', 'figure', allow_duplicate=True),
     Output('latent_pacmap_plot', 'figure', allow_duplicate=True),
     Output('latent_trimap_plot', 'clickData'),
     Output('latent_umap_plot', 'clickData'),
     Output('latent_tsne_plot', 'clickData'),
     Output('latent_pacmap_plot', 'clickData')],
    [Input('latent_trimap_plot', 'clickData'),
     Input('latent_umap_plot', 'clickData'),
     Input('latent_tsne_plot', 'clickData'),
     Input('latent_pacmap_plot', 'clickData')],
    prevent_initial_call=True
)
def display_click_image_latent(trimapClickData, umapClickData, tsneClickData, pacmapClickData):
    clickData = None
    inputs = [trimapClickData, umapClickData, tsneClickData, pacmapClickData]
    for inp in inputs:
        if inp is not None:
            clickData = inp
            break

    if clickData is None:
        raise PreventUpdate
    
    
    original_index = clickData['points'][0]['customdata'][0]

    df_row = df_latent.loc[original_index]
    original_label = df_row['label']
    original_image = df_row['image']

    trimap_img = make_latent_figure(df_latent, 'trimap', index=original_index)
    umap_img = make_latent_figure(df_latent, 'umap', index=original_index)
    tsne_img = make_latent_figure(df_latent, 'tsne', index=original_index)
    pacmap_img = make_latent_figure(df_latent, 'pacmap', index=original_index)

    return original_image, f'Label: {original_label}', original_index, trimap_img, umap_img, tsne_img, pacmap_img, None, None, None, None

@callback(
    [Output('clicked-index-mammoth', 'data'),
     Output('mammoth_original_plot', 'figure', allow_duplicate=True),
     Output('mammoth_trimap_plot', 'figure', allow_duplicate=True),
     Output('mammoth_umap_plot', 'figure', allow_duplicate=True),
     Output('mammoth_tsne_plot', 'figure', allow_duplicate=True),
     Output('mammoth_pacmap_plot', 'figure', allow_duplicate=True),
     Output('mammoth_original_plot', 'clickData'),
     Output('mammoth_trimap_plot', 'clickData'),
     Output('mammoth_umap_plot', 'clickData'),
     Output('mammoth_tsne_plot', 'clickData'),
     Output('mammoth_pacmap_plot', 'clickData'),],
    [Input('mammoth_original_plot', 'clickData'),
     Input('mammoth_trimap_plot', 'clickData'),
     Input('mammoth_umap_plot', 'clickData'),
     Input('mammoth_tsne_plot', 'clickData'),
     Input('mammoth_pacmap_plot', 'clickData')],
    [State('mammoth_original_plot', 'figure'),
     State('mammoth_trimap_plot', 'figure'),
     State('mammoth_umap_plot', 'figure'),
     State('mammoth_tsne_plot', 'figure'),
     State('mammoth_pacmap_plot', 'figure')],
    prevent_initial_call=True
)
def display_click_mammoth(originalClickData, trimapClickData, umapClickData, tsneClickData, pacmapClickData,
                          originalfig, trimapfig, umapfig, tsnefig, pacmapfig):
    clickData = None
    inputs = [originalClickData, trimapClickData, umapClickData, tsneClickData, pacmapClickData]
    
    for inp in inputs:
        if inp is not None:
            clickData = inp
            break

    if clickData is None:
        raise PreventUpdate

    original_index = clickData['points'][0]['customdata'][0]
    data_row = df_mammoth.loc[original_index]

    originalfig['data'] = [
            originalfig['data'][0], go.Scatter3d(
                x=[data_row['x']], 
                y=[data_row['y']], 
                z=[data_row['z']], 
                mode='markers',
                marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='black', line=go.scatter3d.marker.Line(width=5, color='black')),
                showlegend=False
            ).to_plotly_json()
    ]

    # original_img = make_mammoth_figure(df_mammoth, 'original', index=original_index)
    trimapfig['data'] = [
            trimapfig['data'][0], go.Scatter3d(
                x=[data_row['x_trimap_nin_12_nout_4']], 
                y=[data_row['y_trimap_nin_12_nout_4']], 
                z=[data_row['z_trimap_nin_12_nout_4']], 
                mode='markers',
                marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='black', line=go.scatter3d.marker.Line(width=5, color='black')),
                showlegend=False
            ).to_plotly_json()
    ]
    umapfig['data'] = [
            umapfig['data'][0], go.Scatter3d(
                x=[data_row['x_umap_nneighbors_15_mindist_0.1']], 
                y=[data_row['y_umap_nneighbors_15_mindist_0.1']], 
                z=[data_row['z_umap_nneighbors_15_mindist_0.1']], 
                mode='markers',
                marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='black', line=go.scatter3d.marker.Line(width=5, color='black')),
                showlegend=False
            ).to_plotly_json()
    ]
    
    tsnefig['data'] = [
            tsnefig['data'][0], go.Scatter3d(
                x=[data_row['x_tsne_perp_30_exa_12']], 
                y=[data_row['y_tsne_perp_30_exa_12']], 
                z=[data_row['z_tsne_perp_30_exa_12']], 
                mode='markers',
                marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='black', line=go.scatter3d.marker.Line(width=5, color='black')),
                showlegend=False
            ).to_plotly_json()
    ]

    pacmapfig['data'] = [
            pacmapfig['data'][0], go.Scatter3d(
                x=[data_row['x_pacmap_nneighbors_10_init_pca']], 
                y=[data_row['y_pacmap_nneighbors_10_init_pca']], 
                z=[data_row['z_pacmap_nneighbors_10_init_pca']], 
                mode='markers',
                marker=dict(symbol='diamond-open', size=10, opacity=1.0, color='black', line=go.scatter3d.marker.Line(width=5, color='black')),
                showlegend=False
            ).to_plotly_json()
    ]
    
    return original_index, originalfig, trimapfig, umapfig, tsnefig, pacmapfig, None, None, None, None, None
    

@app.callback(
    [#Output('mnist_dd_choose_embedding', 'dropdown_value'),
     Output('mnist_main_plot', 'figure'),
     Output('mnist_subplot_1', 'figure'),
     Output('mnist_subplot_2', 'figure'),
     Output('mnist_subplot_3', 'figure'), 
     Output('sliders_div', 'children')],
    [Input('mnist_dd_choose_embedding', 'value')],
    [State('version_parameters', 'data'),
     State('clicked-index-mnist', 'data'),
     State('versions_on_latent', 'data')]
)
def switch_main_img(dropdown_value, version_parameters, clicked_index, version_on_latent):
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

            html.Div([
                html.Button(get_button_name(version_on_latent[0]), id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '20px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                dbc.Button(
                    html.I(className="bi bi-info-circle-fill me-2"),
                    id="info_button_mnist", 
                    color="info", 
                    className="me-2", 
                    style={"fontSize": "1.5rem", "background-color": "white", "boder-radius": "10px", "border-width": "0px"}
                ),
                dbc.Popover(
                    [
                        dbc.PopoverBody(dcc.Markdown(information_mnist, style={"font-size": "18px", 'color': '#FFFFFF'})),
                    ],
                    target="info_button_mnist",
                    trigger="click",
                    style = {'background-color': '#595959'}
                ),

            ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),

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
            html.Div([
                html.Button(get_button_name(version_on_latent[1]), id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '20px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                dbc.Button(
                    html.I(className="bi bi-info-circle-fill me-2"),
                    id="info_button_mnist", 
                    color="info", 
                    className="me-2", 
                    style={"fontSize": "1.5rem", "background-color": "white", "boder-radius": "10px", "border-width": "0px"}
                ),
                dbc.Popover(
                    [
                        dbc.PopoverBody(dcc.Markdown(information_mnist, style={"font-size": "18px", 'color': '#FFFFFF'})),
                    ],
                    target="info_button_mnist",
                    trigger="click",
                    style = {'background-color': '#595959'}
                ),

            ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),
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
            html.Div([
                html.Button(get_button_name(version_on_latent[2]), id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '20px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                dbc.Button(
                    html.I(className="bi bi-info-circle-fill me-2"),
                    id="info_button_mnist", 
                    color="info", 
                    className="me-2", 
                    style={"fontSize": "1.5rem", "background-color": "white", "boder-radius": "10px", "border-width": "0px"}
                ),
                dbc.Popover(
                    [
                        dbc.PopoverBody(dcc.Markdown(information_mnist, style={"font-size": "18px", 'color': '#FFFFFF'})),
                    ],
                    target="info_button_mnist",
                    trigger="click",
                    style = {'background-color': '#595959'}
                ),

            ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),
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

            html.Div([
                html.Button(get_button_name(version_on_latent[3]), id='translate-button', n_clicks=0, style={'background-color': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '20px', 'margin': '4px 2px', 'border-radius': '12px', 'transition': 'background-color 0.3s ease'}),
                dbc.Button(
                    html.I(className="bi bi-info-circle-fill me-2"),
                    id="info_button_mnist", 
                    color="info", 
                    className="me-2", 
                    style={"fontSize": "1.5rem", "background-color": "white", "boder-radius": "10px", "border-width": "0px"}
                ),
                dbc.Popover(
                    [
                        dbc.PopoverBody(dcc.Markdown(information_mnist, style={"font-size": "18px", 'color': '#FFFFFF'})),
                    ],
                    target="info_button_mnist",
                    trigger="click",
                    style = {'background-color': '#595959'}
                ),

            ], style={'padding': '20px', 'borderRadius': '15px', 'background': '#FFFFFF', 'align-items': 'center', 'display': 'flex', 'justify-content': 'center', 'width': '85%', 'margin': '10px'}),

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