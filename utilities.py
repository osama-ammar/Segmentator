import numpy as np
import numpy.typing as npt
from PIL import Image
import base64
import io
import cv2
import onnxruntime
<<<<<<< HEAD
import random
=======
from scipy.ndimage import binary_dilation
>>>>>>> effe17a4f0c688a13f8773c762cba7c7434a584e

###########################
# helper functions
###########################
def base64_to_array(base64_string, shape=None):
    image_data = base64.b64decode(base64_string)
    image_data = Image.open(io.BytesIO(image_data))
    if shape:
        image_data = image_data.resize(shape)
        print("decoding image with resize ")
    return np.array(image_data)

def numpy_to_base64_image(numpy_image):
    # Convert NumPy array to image (OpenCV style)
    _, buffer = cv2.imencode('.png', numpy_image)
    # Convert to base64 encoding
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_image}"


def image_1d_to_2d(image_1d):
    input_image = np.array(image_1d).reshape(512, 512)
    input_image = np.expand_dims(input_image, axis=2)  # (512, 512) --> (512, 512, 1)
    input_image = np.repeat(input_image, 3, axis=2)  # (512, 512,1) --> (512, 512, 3)
    return input_image


def mask_to_boundary(mask):
    """given a mask we will get boundary points of this mask"""
    # Convert mask to binary image (0 and 255 values)
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Apply morphological gradient which is the difference between dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    points = cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, kernel)
    return points


# prepare input image for model inference
def prepare_model_input(input_image):
    # input_image=np.expand_dims(input_image, axis=2)
    input_image = np.transpose(input_image, (2, 0, 1))
    # Convert to NumPy array and normalize pixel values
    input_image = input_image.astype(np.float32) / 255.0
    # Adjust array values
    input_image -= 0.5
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    # Convert to single-channel image
    input_image = np.mean(input_image, axis=1, keepdims=True)
    return input_image


def model_inference(onnx_model_path, input_array):
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_output = ort_session.run(None, ort_inputs)[0]
    # print(ort_output)
    return ort_output


def show_mask_on_image(input_image, onnx_model_path):
    input_image = prepare_model_input(input_image)
    output_mask = model_inference(onnx_model_path, input_image)
    return output_mask

<<<<<<< HEAD


def get_mask_outlines(mask: npt.NDArray) -> npt.NDArray:
    """given a mask we will get boundary points of this mask  """
    print(mask.shape)
    
    # Convert mask to binary image (0 and 255 values)
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Apply morphological gradient which is the difference between dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask_binary= cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, kernel)
    coordinates = np.where(mask_binary == 255)
    print(coordinates)
    x_coordinates , y_coordinates = coordinates
    points=[]
    for _ in range(30):
        # Generate a random index within the range of the coordinates
        random_index = random.randint(0, len(x_coordinates) - 1)
        
        # Retrieve the x and y coordinates at the random index
        x = x_coordinates[random_index]
        y = y_coordinates[random_index]
        
        # Append the random point to the list
        points.append((x, y))
    return points

# converting path points to svg format , it takes path as  string 
def path_to_svg(path_without_commands):
    # Split the input string into individual coordinate pairs
    coordinate_pairs = path_without_commands.split(',')
    
    # Initialize an empty string to store the SVG path
    svg_path = 'M'
    
    # Iterate through the coordinate pairs
    for i, pair in enumerate(coordinate_pairs):
        # Add 'M' for the first coordinate pair, 'L' for subsequent pairs
        if i > 0:
            svg_path += 'L'
        # Append the coordinate pair to the SVG path
        svg_path += pair.strip()
    
    # Add the 'Z' command to close the path
    svg_path += 'Z'
    
    return svg_path

# converting svg format to path points
def svg_to_path(svg_path):
    # Remove the 'M' command and split the string into coordinate pairs
    coordinate_pairs = svg_path[1:-1].split('L')
    
    # Initialize an empty string to store the path without SVG commands
    path_without_commands = ''
    
    # Iterate through the coordinate pairs
    for pair in coordinate_pairs:
        # Append the coordinate pair to the path without commands, separated by comma
        path_without_commands += pair.strip() + ','
    
    # Remove the trailing comma and return the path without commands
    return path_without_commands[:-1]
=======
def mask_to_edge(mask):
    # Dilate the binary mask
    dilated_mask = binary_dilation(mask,iterations=5)

    # Subtract the original mask to get edges
    edges = dilated_mask.astype(np.uint8) - mask
    return edges


def combined_image_mask(output_mask, image, mode, transperency=150):
    # preprocessing according to output
    if mode == "UNET":
        output_mask = output_mask.reshape(
            (2, 512, 512)
        )  # (1, 2, 512, 512) ---> (2, 512, 512)

        # Compute softmax along the appropriate axis
        output_mask = np.exp(output_mask) / np.sum(
            np.exp(output_mask), axis=0
        )  # ( 2, 512, 512) ---> (2, 512, 512)

        # Find the index of the maximum value along the specified axis
        output_mask = np.argmax(output_mask, axis=0)  # (1, 2, 512, 512) ---> (512, 512)

    if mode == "Mobile_SAM":

        output_mask = output_mask.reshape(output_mask.shape[2], output_mask.shape[3])

    else:
        print(f"using efficient SAM")

    # SAVE MASK HERE
    alpha = output_mask * transperency  # Adjust trasnsparency level
    alpha[alpha == 0] = 255

    # changing only one channel of pixels 0 for red , 1 for green ,  2 for blue
    # we zeroed 2 channels to make the third stand out ( can be done with better method..)
    image[alpha == transperency, 0] = 0
    image[alpha == transperency, 2] = 0
    combined_data = np.stack(
        [image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha],
        axis=2,
    )
    return combined_data
>>>>>>> effe17a4f0c688a13f8773c762cba7c7434a584e
