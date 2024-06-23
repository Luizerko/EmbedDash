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
    "width": 800,
    "height": 640,
    "margin": dict(l=20, r=20, t=70, b=20),
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
            size=18,
            color="black"
        ),
        bgcolor="White"
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
    "margin": dict(l=20, r=20, t=60, b=20),
    "scene": {
        "bgcolor": "rgb(229, 236, 246)",
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
        },
        "camera": {
            'eye': {
                "x" :1.25,
                "y" :1.25,
                "z" :1.25
            },
            'up': {
                "x" :-1,
                "y" :0.25,
                "z" :-1
            },
            'center': {
                "x" :0,
                "y" :0,
                "z" :0
            }
        }
    },
    # "scene_camera":{
    #     "camera": {
    #         "x" :1.25,
    #         "y" :1.25,
    #         "z" :1.25
    #     }
    # },
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
    ),
    "coloraxis_showscale": False
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
    "width": 400,
    "height": 320,
    "margin": dict(l=0, r=0, t=60, b=0),
    "paper_bgcolor": "White",
    "xaxis": dict(showgrid=False, zeroline=False, visible=False),
    "yaxis": dict(showgrid=False, zeroline=False, visible=False),
    "showlegend": False
}