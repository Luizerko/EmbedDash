import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import trimap
from keras.datasets import mnist
import base64
import io
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

app = Dash(__name__)

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

# Paths to cache files
dataframe_path = 'mnist_trimap_dataframe.pkl'
start_time = time.time()

# Check if the DataFrame cache exists
if os.path.exists(dataframe_path):
    # Load the DataFrame from the file
    df = pd.read_pickle(dataframe_path)
else:
    # Load MNIST dataset (e.g., 10% of the data)
    train_examples, train_labels, test_examples, test_labels = load_mnist()
    examples = np.concatenate((train_examples, test_examples), 0)
    base64_images = convert_images_to_base64(examples)
    labels = np.concatenate((train_labels, test_labels))
    indices = np.arange(len(examples))

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

    # Save the DataFrame to a file for future use
    df.to_pickle(dataframe_path)
    df.to_csv('temp.csv')

print("Total processing time: {:.2f} seconds".format(time.time() - start_time))

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
        html.Img(id='hover-image', style={'height': '200px'}),
        html.Div(id='hover-index'),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
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
def display_image(hoverData):
    # display the image below

    if hoverData is None:
        return '', ''
    original_label = hoverData['points'][0]['customdata'][0]
    original_image = hoverData['points'][0]['customdata'][1]
    return original_image, f'Original Label: {original_label}'

if __name__ == '__main__':
    app.run_server(debug=True)
