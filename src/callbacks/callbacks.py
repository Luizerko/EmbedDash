import numpy as np

from dash import Input, Output, State, callback
from utils import (
    sort_coordinates_with_others,
    closest_points_to_line
)


# Define the callback to update the clicked images
CLICKED_POINTS = []
CLICKED_POINTS_INDEX = 0
PREV_CLICKDATA = None


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


# callback for gathering info on the actually clicked points
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
        PREV_CLICKDATA=clickData
    return None


@callback(
    Output('image-find-between-button', 'n_clicks'),
    Input('image-find-between-button', 'n_clicks')
)
def find_images_between_clicks(n_clicks):
    if not n_clicks:
        return None

    global CLICKED_POINTS # TODO: please don't use global variables, it's bad practice
    global CLICKED_POINTS_INDEX # TODO: please don't use global variables, it's bad practice


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


# Callback for the latent space distances function
@callback(
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
