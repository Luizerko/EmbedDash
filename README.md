# Trimap-Vis 
Trimap visualization tool for Multimedia Analytics course at UvA.

The idea of this project is to develop a variety of interactive methods for visualizing embeddings generated by TRIMAP. By examining the data from different perspectives, we can better understand it and start to think about the true relationship between the final visualization and the underlying semantic space. This deeper insight should help you intuitively grasp the data you’re working with, paving the way for new discoveries!

# Environment Installation
For a very standard and straight forward installation proces, open your terminal and run:

```
conda create -n mma python=3.10

git clone https://github.com/Luizerko/mma_trimap_vis.git
cd mma_trimap_vis

pip install -r requirements.txt
```

# Running App
To run the app, simply enter the project folder and run:

```
python3 app.py
```

Then open your browser and navigate to `localhost:8050` to start using the tool.