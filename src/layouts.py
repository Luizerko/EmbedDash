fig_layout_dict = {
        "title": {
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


small_fig_layout_dict = {
    "margin": dict(l=0, r=0, t=0, b=0),
    "paper_bgcolor": "White",
    "xaxis": dict(showgrid=False, zeroline=False, visible=False),
    "yaxis": dict(showgrid=False, zeroline=False, visible=False),
    "showlegend": False
}
