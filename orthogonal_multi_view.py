import json
import imageio
from dash_slicer import VolumeSlicer
from dash import Dash, html, Input, Output, callback

app = Dash(__name__, update_title=None)

vol = imageio.volread("imageio:stent.npz")
slicer0 = VolumeSlicer(app, vol, axis=0)
slicer1 = VolumeSlicer(app, vol, axis=1)
slicer2 = VolumeSlicer(app, vol, axis=2)

slicer0.graph.config["scrollZoom"] = True
slicer1.graph.config["scrollZoom"] = False
slicer2.graph.config["scrollZoom"] = False

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "33% 33% 33%",
    },
    children=[
        html.Div([slicer0.graph, html.Br(), slicer0.slider, *slicer0.stores]),
        html.Div([slicer1.graph, html.Br(), slicer1.slider, *slicer1.stores]),
        html.Div([slicer2.graph, html.Br(), slicer2.slider,html.Div(id='info'), *slicer2.stores]),
    ],
    
)


@callback(
    Output("info", "children"),
    Input(slicer2.state.id, "data")
)
def update_info(state):
    return json.dumps(state, indent=4)




if __name__ == "__main__":
    app.run(debug=True, dev_tools_props_check=False)
