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
import torch
from call_mobile_sam import onnx_process_image
from call_efficient_sam import run_efficient_sam
from utilities import *
import cv2
from dash_canvas import DashCanvas
import threading
"""
NOTES
======
- no of inputs + states in call back function should = no of arguments

"""
# selecting a style
load_figure_template("SUPERHERO")


# pathes and configs
# image_path = "D:\\chest-x-ray.jpeg"
onnx_model_path = "weights\\unet-2v.onnx"
MASK_TRANSPARENCY = 200
mask_corners=0


brush_mask=0
lock=threading.Lock() #A lock is a synchronization primitive that allows only one thread to execute a block of code at a time, while other threads must wait for the lock to be released.


###########################
# Define Dash app
###########################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.config.suppress_callback_exceptions = True
app.config.prevent_initial_callbacks = 'initial_duplicate'
config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ],
    "displaylogo": False,
}

# Helper function to encode an image in base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_image}"


# to be used as default shape in case no segmentation is done
def blank_figure():
    # Creating an empty image array
    color = [50, 50, 50]  # Red color in RGB format
    # Create an empty image with the specified color
    empty_image = np.full(
        (600, 600, 3), color, dtype=np.uint8
    )  # Adjust the size as needed
    fig = px.imshow(empty_image)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(
        template=None,
        width=600,  # Set width to image width
        height=500,  # Set height to image height
    )
    fig.layout.autosize = True

    return fig


def create_button(label, button_id, color):
    return dbc.Button(
        label,
        id=button_id,
        color=color,
        size="sm",  # Smaller button
        className="rounded-button me-2 mt-3",  # Custom class for rounded styling
        n_clicks=0,
        style={"borderRadius": "12px", "padding": "5px 15px"},  # Inline rounded and smaller padding
    )
    
image_card = dbc.Card(
    [
        dbc.CardHeader("Input Image"),
        dbc.CardBody(
            [
                dcc.Graph(
                    id="input_image_id",  # Set an ID for the graph
                    figure=blank_figure(),
                    responsive="auto",
                    style={
                        "width": "100%",
                        "height": "100%",
                        "margin": "auto",
                        "display": "block",
                    },
                    config=config,  # Enable shape editing
                ),


                create_button("Show Mask", "show-mask-button", "primary"),  # Add right margin
                create_button("Mobile SAM", "use-mob-sam", "success"),      # Add right margin
                create_button("Efficient SAM", "use-eff-sam", "info"),                     # No margin needed on the last button
            ]
        ),


        ############################################################
        dbc.CardFooter(
            [
                html.H6("import an X-Ray image and press show mask"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
        ###############################################################################
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "90%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploade
            multiple=True,
            
        ),

        dbc.CardFooter(
            id="canava-image",
            children=html.Div([
            DashCanvas(id='canvaas_image',
                    lineWidth=10,
                    lineColor='rgba(0, 255, 0, 0.7)',
                    filename=encode_image('dental.jpg'),
                    width=235,         # Match canvas width to image width
                    height=215,       # Match canvas height to image height
                    goButtonTitle="Save",
                    tool='draw', 
                    hide_buttons =["zoom", "pan", "line","rectangle", "undo", "select"] ,                  
                    )])

        ),
            # Button to trigger mask processing
            html.Button("Process Mask", id="process-btn", n_clicks=0),

            # Output display
            html.Div(id='output-data', style={"marginTop": "20px", "textAlign": "center"}),


    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={"width": "100%", "height": "auto", "margin": "auto"},
)


# Define the mask card layout
mask_image_card = dbc.Card(
    [
        dbc.CardHeader("Image with mask"),
        dbc.CardBody(
            [
                dcc.Graph(
                    id="mask_image_id",  # Set an ID for the graph
                    figure=blank_figure(),
                    responsive="auto",
                    config=config,  # Enable shape editing
                    style={
                        "width": "100%",
                        "height": "100%",
                        "margin": "auto",
                        "display": "block",
                    },
                ),
  
                create_button("Show Annotation info", "#print-annotation", "info"),
                
                # Hidden div to store annotations data
                html.Div(id="annotations-data", style={"display": "none"}),
                html.Div(id="#print-output"),
                ##########################################################################
            ]
        ),
        dbc.CardFooter(
            [
                html.H6("Chest X-Ray segmented image"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={"width": "100%", "height": "auto", "margin": "auto"},
)


####################
# Define main app layout
####################
app.layout = html.Div(
    [
        dbc.Row(
            [dbc.Col(image_card, width=5), dbc.Col(mask_image_card, width=5)],
            justify="center",  # This will center the columns horizontally
            style={"margin": "auto", "padding": "20px"},  # Optional padding for better spacing         
            ),
    ], 
)


#############
# Callbacks
##############
# Callback to process and retrieve mask data
@app.callback(
    Output('canvaas_image', 'image_content'),#, allow_duplicate=True),
    #Output("canvaas_image", "image_content"),
    Input('process-btn', 'n_clicks'),
    Input('canvaas_image', 'json_data'),
    prevent_initial_call=True,

)
def brush_to_mask(n_clicks, json_data):
    mask_corners = [70, 70, 300, 300]
    
    if n_clicks > 0 and json_data:
        # Parse the JSON data from DashCanvas
        data = json.loads(json_data)
        input_image_64 = data['objects'][0]['src']
        input_image_64=input_image_64.split(",")[1]+"="
        global brush_mask
        # Iterate over paths in the JSON data
        for shape in data['objects']:

            if shape['type'] == 'path':  # Check if the object is a path
                #print(input_image_64)
                
                path = shape['path']
                x_points = []
                y_points = []
                for cmd in path:
                    # getting start , in between , end  points in the mask
                    if cmd[0] in ['M',"Q", 'L']:  # Move or Line commands
                        x, y = cmd[1], cmd[2]
                        x_points.append(int(x))
                        y_points.append(int(y))
                        
                # get min and max corner
                min_x , max_x = min(x_points) , max(x_points)
                min_y , max_y = min(y_points) , max(y_points)
                mask_corners = [min_x, min_y, max_x, max_y]

                input_image = base64_to_array(input_image_64)
                
                # print(f"brush image shape : {input_image.shape}")
                # print(f"mask_corners : {mask_corners} ,{y_points}")
                
                input_box = np.array(mask_corners).astype(np.int32)

                output_masks = run_efficient_sam(input_image, input_box)
                #output_masks=mask_to_edge(output_masks)

                #with lock: # Ensures only one thread can modify the global var at a time
                brush_mask =np.logical_or(brush_mask, output_masks).astype(np.uint8)
                print(f"brush_mask shape input {brush_mask.shape} , {np.max(brush_mask)}")
                
                    
                # blending image and mask
                combined_data = combined_image_mask(
                    brush_mask, input_image, mode="eff-sam", transperency=MASK_TRANSPARENCY
                )
                print(f"efficient sam shape output {combined_data.shape} , {np.max(combined_data)}")
                
                encoded_combined_data = numpy_to_base64_image(combined_data)


                return encoded_combined_data

# Define callback to #print shapes(rectangles , lassos...) when the button is clicked
@app.callback(
    Output("#print-output", "children"),
    [Input("input_image_id", "relayoutData"), Input("#print-annotation", "n_clicks")],
    prevent_initial_call=True,
)
def get_annotations_data(relayout_data, n_clicks):
    if n_clicks > 0:
        if "shapes" in relayout_data:
            output_json = json.dumps(relayout_data["shapes"], indent=2)
            if len(relayout_data["shapes"]) > 0:
                [x, y] = [
                    relayout_data["shapes"][0]["x0"],
                    relayout_data["shapes"][0]["y0"],
                ]
                # print([x, y])
            return output_json
    else:
        return no_update


# Callback to update mask overlay when button is clicked
@app.callback(
    Output("mask_image_id", "figure", allow_duplicate=True),
    [Input("show-mask-button", "n_clicks")],
    [State("input_image_id", "figure")],
    prevent_initial_call="initial_duplicate",
)
def show_chest_UNET(n_clicks, current_figure):
    if n_clicks > 0:
        # images in plotly dash figure in ~ 2 formats as follows ....1D format or 64-encoded (image encoded as text)
        if "z" in current_figure["data"][0].keys():
            # print("using normal image ")
            input_image = image_1d_to_2d(current_figure["data"][0]["z"])
            output_mask = show_mask_on_image(
                input_image, onnx_model_path
            )  # (1,2,512, 512)

        else:
            print("using 64-encoded image ")
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure["data"][0]["source"][22:-1] + "=",shape=(512, 512)
            )
            #input_image.resize((512, 512, 3), refcheck=False)
            
            # Update the figure data with mask overlay
            output_mask = show_mask_on_image(input_image, onnx_model_path)

        # blending image and mask
        combined_data = combined_image_mask(
            output_mask, input_image, mode="UNET", transperency=MASK_TRANSPARENCY
        )
        print(combined_data.shape)

        updated_figure = px.imshow(
            combined_data,
            zmin=0,
            zmax=255,
            color_continuous_scale="green",  # Example color scale
            labels={"color": "Heatmap Value"},
        )
        return updated_figure

    else:
        return current_figure


# Callback to use mobile SAM model for segmentation with Box prompt
@app.callback(
    Output("mask_image_id", "figure", allow_duplicate=True),
    [Input("use-mob-sam", "n_clicks")],
    [State("input_image_id", "figure"), State("input_image_id", "relayoutData")],
    prevent_initial_call=True,
)
def show_mob_sam_mask(n_clicks, current_figure, relayout_data):
    # default values for mobile SAM point and boxes

    if n_clicks > 0:
        input_point = np.array([[300, 350]])
        input_box = np.array([200, 200, 300, 300])
        input_label = np.array([1])

        # check if shape is drawn
        if relayout_data != None and "shapes" in relayout_data:
            [x1, x2, y1, y2] = [
                relayout_data["shapes"][0]["x0"],
                relayout_data["shapes"][0]["y0"],
                relayout_data["shapes"][0]["x1"],
                relayout_data["shapes"][0]["y1"],
            ]
            input_point = np.array([[x1, y1]]).astype(np.int32)

        # check if shape is updated
        if relayout_data != None and "shapes[0].x0" in relayout_data:
            [x1, x2, y1, y2] = relayout_data.values()
            input_point = np.array([[x1, y1]]).astype(np.int32)
            
            
        [min_x, min_y, max_x, max_y] = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]
        input_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)

        # 'z' key here carries image info in plotly dash figure
        if "z" in current_figure["data"][0].keys():
            # print("using normal image ")
            # mostly one channel images in a list format
            input_image = image_1d_to_2d(current_figure["data"][0]["z"])

        else:
            #print(current_figure["data"][0]["source"][22:-1] + "=")
            # print("using 64-encoded image ")
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure["data"][0]["source"][22:-1] + "="
            )

        output_masks = onnx_process_image(
            input_image.astype(np.float32),
            input_point,
            input_box=input_box,
            input_label=input_label,
        )

        output_masks=mask_to_edge(output_masks)
        # blending image and mask
        combined_data = combined_image_mask(
            output_masks, input_image, mode="Mobile_SAM", transperency=MASK_TRANSPARENCY
        )
        
        print(combined_data.shape)
        updated_figure = px.imshow(
            combined_data,
            zmin=0,
            zmax=255,
            color_continuous_scale="green",  # Example color scale
        )

        return updated_figure


# Callback to use Efficient SAM model for segmentation with Box prompt
@app.callback(
    Output("mask_image_id", "figure", allow_duplicate=True),
    [Input("use-eff-sam", "n_clicks")],
    [State("input_image_id", "figure"), State("input_image_id", "relayoutData")],
    prevent_initial_call=True,
)
def show_eff_sam_mask(n_clicks, current_figure, relayout_data):
    # default values for mobile SAM point and boxes

    if n_clicks > 0:
        input_box = np.array([200, 200, 300, 300])
        # check if shape is drawn
        if relayout_data != None and "shapes" in relayout_data: 
            [x1, x2, y1, y2] = [
                relayout_data["shapes"][0]["x0"],
                relayout_data["shapes"][0]["y0"],
                relayout_data["shapes"][0]["x1"],
                relayout_data["shapes"][0]["y1"],
            ]

        # check if shape is updated
        if relayout_data != None and "shapes[0].x0" in relayout_data:
            [x1, x2, y1, y2] = relayout_data.values()



        [min_x, min_y, max_x, max_y] = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]
        input_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)


        # 'z' key here carries image info in plotly dash figure
        if "z" in current_figure["data"][0].keys():
            # mostly one channel images in a list format
            input_image = image_1d_to_2d(current_figure["data"][0]["z"])

        else:
            # print("using 64-encoded image ")
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure["data"][0]["source"][22:-1] + "="
            )

        output_masks = run_efficient_sam(input_image, input_box)
        output_masks=mask_to_edge(output_masks)

        # blending image and mask
        combined_data = combined_image_mask(
            output_masks, input_image, mode="eff-sam", transperency=MASK_TRANSPARENCY
        )
        print("combined_data ........................................ ")
        print(combined_data.shape)
        
        updated_figure = px.imshow(
            combined_data,
            zmin=0,
            zmax=255,
            color_continuous_scale="green",  # Example color scale
        )

        return updated_figure


# Callback to update figure when uploading new image


@app.callback(
    Output("input_image_id", "figure"),
    [Input("upload-image", "contents")],
    [State("input_image_id", "figure")],
)
def upload_image(list_of_contents, current_figure):
    if list_of_contents is not None:
        _, base64_string = list_of_contents[0].split(",")
        image = base64_to_array(base64_string)#, shape=(512, 512))
        # print(image.shape)

        updated_figure = px.imshow(
            image,
            zmin=0,
            zmax=255,
            # height = image.size[0],
            # width = image.size[1],
            color_continuous_scale="gray",  # Example color scale
            labels={"color": "Heatmap Value"},
        )

        return updated_figure
    else:
        return current_figure


###################
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
