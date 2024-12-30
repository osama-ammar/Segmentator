"""
An example demonstrating adding traces.

This shows a volume with contours overlaid on top. The `extra_traces`
property is used to add scatter traces that represent the contours.
"""

import plotly
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from dash_slicer import VolumeSlicer
import imageio
from skimage import measure
import numpy as np

app = dash.Dash(__name__, update_title=None)
server = app.server

vol = imageio.volread("imageio:stent.npz")


def normalize_vol(vol,wl,ww):
    lower_limit = wl-(ww/2)
    upper_limit = wl+(ww/2)
    
    # Step 1: Clip the values to be within the range [100, 600]
    clipped_vol = np.clip(vol, lower_limit, upper_limit)
    # Step 2: Normalize the volume to the range [0, 1]
    vol = (clipped_vol - clipped_vol.min()) / (clipped_vol.max() - clipped_vol.min()) *255
    print(vol.shape ,vol.max())
    return vol

vol = normalize_vol(vol,50,200)



mi, ma = vol.min(), vol.max()
slicer = VolumeSlicer(app, vol, clim=(0, 800))

app.layout = html.Div(
    [
        slicer.graph,
        slicer.slider,
        dcc.RangeSlider(
            id="level-slider",
            min=vol.min(),
            max=vol.max(),
            step=1,
            value=[mi + 0.1 * (ma - mi), mi + 0.3 * (ma - mi)],
        ),
        *slicer.stores,
    ]
)

# Define colormap to make the lower threshold shown in yellow, and higher in red
colormap = [(255, 255, 0, 50), (255, 0, 0, 100)]


@app.callback(
    Output(slicer.overlay_data.id, "data"),
    [Input("level-slider", "value")],
)
def apply_levels(level):
    mask = np.zeros(vol.shape, np.uint8)
    mask += vol > level[0]
    mask += vol > level[1]
    print(mask.shape)
    return slicer.create_overlay_data(mask, colormap)


if __name__ == "__main__":
    # Note: dev_tools_props_check negatively affects the performance of VolumeSlicer
    app.run_server(debug=True, dev_tools_props_check=False)