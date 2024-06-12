fig_layout_dict = {
        "title": {
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 32,
            'color': 'black',
            'family': 'Arial Black', 
            'weight': 'bold'
        }
    },
    "margin": dict(l=20, r=20, t=100, b=20),
    "paper_bgcolor": "White",
    "xaxis": dict(showgrid=False, zeroline=False, visible=False),
    "yaxis": dict(showgrid=False, zeroline=False, visible=False),
    "legend": dict(
        orientation='h',
        x=0.5,
        y=-0.05,
        xanchor='center',
        yanchor='top',
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

fig_layout_dict_mammoth = {
    "title": {
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 32,
            'color': 'black',
            'family': 'Arial Black', 
            'weight': 'bold'
        }
    },
    "margin": dict(l=20, r=20, t=100, b=20),
    "paper_bgcolor": "White",
    "scene": {
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "visible": False
        },
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "visible": False
        },
        "zaxis": {
            "showgrid": False,
            "zeroline": False, 
            "visible": False
        }
    },
    "legend": dict(
        orientation='h',
        x=0.5,
        y=-0.05,
        xanchor='center',
        yanchor='top',
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


small_fig_layout_dict = {
    "title": {
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 32,
            'color': 'black',
            'family': 'Arial Black', 
            'weight': 'bold'
        }
    },
    "margin": dict(l=0, r=0, t=50, b=0),
    "paper_bgcolor": "White",
    "xaxis": dict(showgrid=False, zeroline=False, visible=False),
    "yaxis": dict(showgrid=False, zeroline=False, visible=False),
    "showlegend": False
}
