import numpy as np
import cv2
from mobile_sam import sam_model_registry, SamPredictor
import onnxruntime


checkpoint = "weights\\mobile_sam.pt"
model_type = "vit_t"
onnx_model_path = "weights\\sam_onnx_example.onnx"
image = cv2.imread("chest-x-ray.jpeg")

sam = sam_model_registry[model_type](checkpoint=checkpoint)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_label = np.array([0])
onnx_box_labels = np.array([2, 3])


def onnx_process_image(image, input_point, input_box=None, input_label=None):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    image_embedding = predictor.get_image_embedding().cpu().numpy()
    if input_box.all()!=None:
        print("mob-SAM box mode")
        onnx_box_coords = input_box.reshape(2, 2)
        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
        onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    else:
        print("mob-SAM point mode")
        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(
            onnx_coord, image.shape[:2]).astype(np.float32)



    # Create an empty mask input and an indicator for no mask.
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # Package the inputs to run in the onnx model
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    return masks
