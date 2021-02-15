import IPython
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
from pycocotools.coco import COCO

dataDir = 'coco'
dataType = 'val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# Don't care about supercategories
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['car']) # This gives the numerical ID of the category [3]
imgIds = coco.getImgIds(catIds=catIds)  # This gives the image filename, height, width, and id (basically the same as filename)

imgs = coco.loadImgs(imgIds) # See example properties below
'''
{'license': 3,
  'file_name': '000000511999.jpg',
  'coco_url': 'http://images.cocodataset.org/val2017/000000511999.jpg',
  'height': 446,
  'width': 640,
  'date_captured': '2013-11-17 10:43:04',
  'flickr_url': 'http://farm9.staticflickr.com/8246/8647068737_1f58d52a62_z.jpg',
  'id': 511999}]
'''

# Select a single image to display bounding box
val_image_path = 'D:/Desktop/EE113DB/Custom Model/coco/images/val2017'
img_filename = os.path.join(val_image_path, imgs[0]['file_name'])
I = io.imread(img_filename)
plt.axis('off')
plt.imshow(I)
annIds = coco.getAnnIds(imgIds=imgs[0]['id'], catIds=[], areaRng=[], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns, draw_bbox=True)
plt.show()


# Pretty Print of JSON data
json_formatted_str = json.dumps(anns, indent=2)
print(json_formatted_str)

# TODO - from this -> Workflow for creating training set
# 1 - Get all images with cars in them
# 2 - Use JSON libarary to parse annotations and pick out JUST the car annotations (category_id == 3) (This is just working with dictionaries on a large scale)
# 3 - Bounding box combined with image resizing code -> (128x128x3)

# 4 - Create JUST A CAR DETECTOR -> Modify network architecture accordingly
# 5 - Custom Training Code/Loss function? See experiencor code on Github

# 5 - Evaluation code? - > Will Have to be Custom
# 6 - Talk to TAs about X-CUBE AI not working -> Saves us a TON of time (talk to other group?)
# 7 - Ask Jamie to implement custom layers from scratch (MP1) -> Leaky ReLU, depthwiseConv2D, BatchNorm, Add, Concatenate, Upsampling2D, etc.



'''
annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}

categories[{
"id": int, "name": str, "supercategory": str,
}]
'''

