import os
import dash
import dash_html_components as html

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


vtk_view = dash_vtk.View(
    dash_vtk.VolumeRepresentation(
        children=[
            dash_vtk.VolumeController(),
            dash_vtk.Volume(state=volume_state),
        ]
    )
)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"height": "calc(100vh - 16px)", "width": "100%"},
    children=[html.Div(vtk_view, style={"height": "100%", "width": "100%"})],
)

if __name__ == "__main__":
    app.run_server(debug=True)