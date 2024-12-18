# Segmentator
- Simple Image segmentation app that includes :
    - UNET for Chest s X-Ray segmentation
    - Mobile SAM ( a light  version of SAM )
    - Efficient SAM (more accurate and faster)
    - 3D Segmentation utilities (coming soon..)

## How to use

- download the repo
- move to project folder and install requirements `pip install requirements.txt`
- run `python main.py`
- open http://127.0.0.1:8050/ in browser
- upload image from local

### using simple chest UNET
- Drag| upload image
- click `Show Mask`
- you will find a mask for this region

![](using-unet.png)

### using Mobile SAM - box prompt
- select rectangle drawer
- draw a bbox
- click `mobile SAM`
- you will find a mask for this region
- edit the box and re-click to update the mask

![](using-mobile-sam.png)

### TODO:
- [x] view image and mask
- [x] import images
- [ ] Handling all images inputs (png , jpg , npy....)
- [ ] editable mask to control points
- [ ] mask to control points
- [ ] include Med SAM
- [ ] allow multible images segmentation - like annotator
- [ ] allow 3D segmentation by 3D UNET
- [ ] allow 3D Medical image segmentation by Med SAM
- [ ] show 3D segmentation 
