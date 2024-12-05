from EfficientSAM.efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)

# from squeeze_sam.build_squeeze_sam import build_squeeze_sam
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile


models = {}

# Build the EfficientSAM-Ti model.
models["efficientsam_ti"] = build_efficient_sam_vitt()

# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("D:/Code_store/segmentator/Segmentator/EfficientSAM/weights/efficient_sam_vits.pt.zip", "r") as zip_ref:
    zip_ref.extractall("weights")
    
# Build the EfficientSAM-S model.
#models["efficientsam_s"] = build_efficient_sam_vits()

# Build the SqueezeSAM model.
# models['squeeze_sam'] = build_squeeze_sam()
bbox_cordinates = torch.tensor([[[[580, 350], [650, 350]]]])

sample_image_np = np.array(Image.open("chest-x-ray-2.jpg"))
print(f"efficient sam shape input {sample_image_np.shape}")
def run_efficient_sam(sample_image_np, bbox_cordinates):
    # load an image
    sample_image_tensor = transforms.ToTensor()(sample_image_np)
    # Feed a few (x,y) points in the mask as input.
    bbox_cordinates = torch.reshape(torch.tensor(bbox_cordinates), [1, 1, -1, 2])
    
    pts_labels = np.array([2,3])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    print(bbox_cordinates)

    # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
    efficient_sam_model = models["efficientsam_ti"]
    model_name = "efficientsam_ti"


    print("Running inference using ", model_name)
    predicted_logits, predicted_iou = efficient_sam_model(
        sample_image_tensor[None, ...],
        bbox_cordinates,
        pts_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:, :, None]
    print(f"efficient sam shape output {mask.shape}")
    
    Image.fromarray(masked_image_np).save(f"dogs_{model_name}_mask.png")
    return mask


#run_efficient_sam(sample_image_np, bbox_cordinates)
