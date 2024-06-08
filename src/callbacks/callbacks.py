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
    Input('image-find-between-button', 'n_clicks'),
    State('scatter-plot', 'figure')
)
def find_images_between_clicks(n_clicks, fig):
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
