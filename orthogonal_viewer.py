import os
import random

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import dash_vtk
from dash_vtk.utils import to_volume_state
import numpy as np
import vtk






def numpy_to_vtk_image(npy_file):
    """Convert a NumPy 3D array to VTK ImageData"""
    # Load NumPy volume
    volume = np.load(npy_file)

    # Ensure it's in the correct format (float32, C-contiguous)
    volume = np.ascontiguousarray(volume, dtype=np.float32)

    # Get the shape (Z, Y, X)
    depth, height, width = volume.shape

    # Convert NumPy array to VTK array
    vtk_array = vtk.vtkFloatArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetNumberOfTuples(depth * height * width)
    
    # Flatten and insert into VTK array
    flat_data = volume.flatten()
    for i in range(len(flat_data)):
        vtk_array.SetValue(i, flat_data[i])

    # Create VTK ImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, depth)
    image_data.SetSpacing(1.0, 1.0, 1.0)  # Adjust if needed
    image_data.SetOrigin(0, 0, 0)

    # Set Scalars (Voxel values)
    image_data.GetPointData().SetScalars(vtk_array)

    return image_data





# Example usage
npy_file = "D:/Datasets/131056.image.npy"  # Replace with your actual file
vtk_image = numpy_to_vtk_image(npy_file)
volume_state = to_volume_state(vtk_image)




sliders = {
    "Slice i": dcc.Slider(id="slider-i", min=0, max=256, value=128),
    "Slice j": dcc.Slider(id="slider-j", min=0, max=256, value=128),
    "Slice k": dcc.Slider(id="slider-k", min=0, max=95, value=47),
    "Color Level": dcc.Slider(id="slider-lvl", min=0, max=4095, value=1000),
    "Color Window": dcc.Slider(id="slider-window", min=0, max=4095, value=4095),
}

controls = dbc.Card(
    body=True,
    children=dbc.Row(
        [
            dbc.Col([dbc.Label(label), component], style={"width": "150px"})
            for label, component in sliders.items()
        ]
    ),
)

slice_property = {"colorWindow": 4095, "colorLevel": 1000}

slice_view = dash_vtk.View(
    id="slice-view",
    cameraPosition=[1, 0, 0],
    cameraViewUp=[0, 0, -1],
    cameraParallelProjection=False,
    background=[0.9, 0.9, 1],
    children=[
        dash_vtk.ShareDataSet(dash_vtk.Volume(state=volume_state)),
        dash_vtk.SliceRepresentation(
            id="slice-repr-i",
            iSlice=128,
            property=slice_property,
            children=dash_vtk.ShareDataSet(),
        ),
        dash_vtk.SliceRepresentation(
            id="slice-repr-j",
            jSlice=128,
            property=slice_property,
            children=dash_vtk.ShareDataSet(),
        ),
        dash_vtk.SliceRepresentation(
            id="slice-repr-k",
            kSlice=47,
            property=slice_property,
            children=dash_vtk.ShareDataSet(),
        ),
    ],
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    fluid=True,
    style={"height": "calc(100vh - 30px)"},
    children=[
        html.Div(
            style={"height": "20%", "display": "flex", "alignItems": "center"},
            children=[
                html.Br(),
                controls,
                html.Br(),
            ],
        ),
        html.Div(slice_view, style={"height": "80%"}),
    ],
)


@app.callback(
    [
        Output("slice-view", "triggerRender"),
        Output("slice-repr-i", "property"),
        Output("slice-repr-i", "iSlice"),
        Output("slice-repr-j", "property"),
        Output("slice-repr-j", "jSlice"),
        Output("slice-repr-k", "property"),
        Output("slice-repr-k", "kSlice"),
    ],
    [
        Input("slider-i", "value"),
        Input("slider-j", "value"),
        Input("slider-k", "value"),
        Input("slider-lvl", "value"),
        Input("slider-window", "value"),
    ],
)
def update_slice_property(i, j, k, level, window):
    render_call = random.random()
    slice_prop = {"colorLevel": level, "colorWindow": window}
    return render_call, slice_prop, i, slice_prop, j, slice_prop, k


if __name__ == "__main__":
    app.run_server(debug=True)