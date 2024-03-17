import dash
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, Input, Output, no_update, callback
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
import plotly.express as px

import json
import numpy as np
from PIL import Image

from call_mobile_sam import onnx_process_image
from utilities import *

# selecting a style
load_figure_template('SUPERHERO')


"""
TODO List:
-----------
[x] - view image and mask
[x] - import images
[/] - understand plotly very well
[/] - view every part of the mask with different color 
[/] - get good layout for portofolio
[ ] - Handling all images inputs (png , jpg , npy....)
[ ] - clean code (remove unneeded code - organize and document)
[ ] - get the area of mask segments and view it 
"""

# pathes and configs
#image_path = "D:\\chest-x-ray.jpeg"
onnx_model_path = "unet-2v.onnx"
mask_trasnsparency = 150


###########################
# Define Dash app
###########################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ],
    'displaylogo': False
}

# to be used as default shape in case noe segmentation is done


def blank_figure():
    # Creating an empty image array
    color = [50, 50, 50]  # Red color in RGB format
    # Create an empty image with the specified color
    empty_image = np.full((600, 600, 3), color, dtype=np.uint8)  # Adjust the size as needed
    fig = px.imshow(empty_image)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(
        template=None,
        width=600,   # Set width to image width
        height=500,  # Set height to image height
    )
    fig.layout.autosize = True

    return fig


image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray Image"),
        dbc.CardBody([
            dcc.Graph(
                id='input_image_id',  # Set an ID for the graph
                figure=blank_figure(),
                responsive='auto',
                style={'width': '100%',
                       'height': '100%',
                       'margin': '5px',
                       'display': 'block'
                       },  # Center the image

                config=config  # Enable shape editing
            ),
        ]),

        ###########################################################
        dbc.Button("Show Mask", id="show-mask-button",
                   color="primary", className="mr-1", n_clicks=0),

        dbc.Button("Mobile SAM", id="use-sam",
                   color="secondary", className="mr-1", n_clicks=0),
        ############################################################
        dbc.CardFooter(
            [
                html.H6(
                    "import an X-Ray image and press show mask"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
        ###############################################################################
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '90%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True),
        html.Div(id='output-image-upload')
    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={'width': '100%', 'height': 'auto', 'margin': 'auto'}

)


# Define the mask card layout
mask_image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray mask"),
        dbc.CardBody([
            dcc.Graph(
                id='mask_image_id',  # Set an ID for the graph
                figure=blank_figure(),
                responsive='auto',
                config=config,  # Enable shape editing
                style={'width': '100%',
                       'height': '100%',
                       'margin': 'auto',
                       'display': 'block'}
            ),

            dbc.Button("#Print Annotations",
                       id="#print-annotation",
                       color="primary",
                       className="mr-1",
                       n_clicks=0),

            # Hidden div to store annotations data
            html.Div(id="annotations-data", style={'display': 'none'}),
            html.Div(id="#print-output")
            ##########################################################################
        ]),
        dbc.CardFooter(
            [
                html.H6(
                    "Chest X-Ray segmented image"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),


    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={'width': '100%', 'height': 'auto', 'margin': 'auto'}
)


####################
# Define main app layout
####################
app.layout = html.Div(
    [
        dbc.Row([dbc.Col(image_card, width=5),
                dbc.Col(mask_image_card, width=5)]),
    ]
)


#############
# Callbacks
##############

# Define callback to #print shapes(rectangles , lassos...) when the button is clicked
@app.callback(
    Output("#print-output", "children"),
    [Input("input_image_id", "relayoutData"),
     Input("#print-annotation", "n_clicks")],
    prevent_initial_call=True,
)
def get_annotations_data(relayout_data, n_clicks):
    if n_clicks > 0:
        if "shapes" in relayout_data:
            output_json = json.dumps(relayout_data["shapes"], indent=2)
            if len(relayout_data["shapes"]) > 0:
                [x, y] = [relayout_data["shapes"][0]['x0'],
                          relayout_data["shapes"][0]['y0']]
                #print([x, y])
            return output_json
    else:
        return no_update


# Callback to update mask overlay when button is clicked
@app.callback(
    Output('mask_image_id', 'figure', allow_duplicate=True),
    [Input('show-mask-button', 'n_clicks')],
    [State('input_image_id', 'figure')],
    prevent_initial_call='initial_duplicate'
)
def update_mask_overlay(n_clicks, current_figure):
    if n_clicks > 0:
        # images in plotly dash figure in ~ 2 formats as follows ....1D format or 64-encoded (image encoded as text)
        if 'z' in current_figure['data'][0].keys():
            #print("using normal image ")
            input_image = image_1d_to_2d(current_figure['data'][0]['z'])
            output_mask = show_mask_on_image(
                input_image, onnx_model_path)  # (1,2,512, 512)

        else:
            #print("using 64-encoded image ")
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure['data'][0]['source'][22:-1]+"=")
            input_image.resize((512, 512, 3), refcheck=False)
            # Update the figure data with mask overlay
            output_mask = show_mask_on_image(input_image, onnx_model_path)

        #print(f"input_image : {input_image.shape} , {output_mask.shape}")
        output_mask = output_mask.reshape((2, 512, 512))

        # Compute softmax along the appropriate axis
        output_mask = np.exp(output_mask) / np.sum(np.exp(output_mask), axis=0)

        # Find the index of the maximum value along the specified axis
        output_mask = np.argmax(output_mask, axis=0)
        alpha = output_mask*mask_trasnsparency  # Adjust trasnsparency level
        alpha[alpha == 0] = 255

        # changing only one channel of pixels 0 for red , 1 for green ,  2 for blue
        # we zeroed 2 channels to make the third stand out ( can be done with better method..)
        input_image[alpha == mask_trasnsparency, 0] = 0
        input_image[alpha == mask_trasnsparency, 2] = 0
        combined_data = np.stack(
            [input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2], alpha], axis=2)
        #print(f"combined : {combined_data.shape} ,{input_image.shape} , {output_mask.shape}  ")

        updated_figure = px.imshow(combined_data,
                                   zmin=0, zmax=255,
                                   color_continuous_scale='green',  # Example color scale
                                   labels={'color': 'Heatmap Value'})
        return updated_figure

    else:
        return current_figure


# Callback to use mobile SAM model for segmentation with Box prompt
@app.callback(
    Output('mask_image_id', 'figure'),
    [Input('use-sam', 'n_clicks')],
    [State('input_image_id', 'figure'), State(
        "input_image_id", "relayoutData")],
    prevent_initial_call=True
)
def show_sam_mask(n_clicks, current_figure, relayout_data):
    # default values for mobile SAM point and boxes

    if n_clicks > 0:
        input_point = np.array([[300, 350]])
        input_box = np.array([200, 200, 300, 300])
        input_label = np.array([1])

        # check if shape is drawn
        if relayout_data != None and "shapes" in relayout_data:
            [x1, x2, y1, y2] = [relayout_data["shapes"][0]['x0'],
                                relayout_data["shapes"][0]['y0'],
                                relayout_data["shapes"][0]['x1'],
                                relayout_data["shapes"][0]['y1']]
            input_point = np.array([[x1, y1]]).astype(np.int32)
            [min_x, min_y, max_x, max_y] = [
                min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            input_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)

        # check if shape is updated
        if relayout_data != None and 'shapes[0].x0' in relayout_data:
            [x1, x2, y1, y2] = relayout_data.values()
            input_point = np.array([[x1, y1]]).astype(np.int32)
            [min_x, min_y, max_x, max_y] = [
                min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            input_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)

        # 'z' key here carries image info in plotly dash figure
        if 'z' in current_figure['data'][0].keys():
            #print("using normal image ")
            # mostly one channel images in a list format
            input_image = image_1d_to_2d(current_figure['data'][0]['z'])
            masks = onnx_process_image(input_image.astype(
                np.float32), input_point, input_box=input_box, input_label=input_label)

        else:
            #print("using 64-encoded image ")
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure['data'][0]['source'][22:-1]+"=")

            masks = onnx_process_image(input_image.astype(
                np.float32), input_point, input_box=input_box, input_label=input_label)

        print(masks.shape)
        color = np.array([255, 0, 0, mask_trasnsparency])
        masks = masks.reshape(masks.shape[2], masks.shape[3], 1) * color.reshape(1, 1, -1)
        print(color.reshape(1, 1, -1).shape)
        
        #masks=np.argmax(masks, axis=2)
        masks[masks == 0] = 255

        combined_data = np.stack(
            [input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2], masks[:, :, 3]], axis=2)

        # print(f"combined : {combined_data.shape} ,{input_image.shape} , {masks.shape},{np.unique(masks)}  ")
        # print(f" default :point {input_point} , input box {input_box}")

        updated_figure = px.imshow(combined_data,
                                   zmin=0, zmax=255,
                                   )

        return updated_figure

# Callback to update figure when uploading new image


@app.callback(
    Output('input_image_id', 'figure'),
    [Input('upload-image', 'contents')],
    [State('input_image_id', 'figure')]
)
def upload_image(list_of_contents, current_figure):
    if list_of_contents is not None:
        _, base64_string = list_of_contents[0].split(',')
        image = base64_to_array(base64_string, shape=(512, 512))
        # print(image.shape)

        updated_figure = px.imshow(image,
                                   zmin=0, zmax=255,
                                   #height = image.size[0],
                                   #width = image.size[1],
                                   color_continuous_scale='gray',  # Example color scale
                                   labels={'color': 'Heatmap Value'})

        return updated_figure
    else:
        return current_figure


###################
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
