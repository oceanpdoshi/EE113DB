import IPython
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
from skimage import img_as_ubyte
from skimage.transform import resize
from pycocotools.coco import COCO
from progressbar import progressbar

dataDir = 'coco'
outDir = 'custom_datasets200'
DataType = 'val2017' # trani2017 - once ready to run on full dataset
annFile='{}/annotations/instances_{}.json'.format(dataDir, DataType)

# Desired image size and categories
# TODO - run this once network architecture finalized
new_w, new_h = 200, 200
catNms = ['car']
area_suppression_factor = 25*25 # (pixels^2) : qualitatively, this seems like a good threshold for 200x200 images
coco = COCO(annFile)

# cats = coco.loadCats(coco.getCatIds())

catIds = coco.getCatIds(catNms=catNms) # 3
imgIds = coco.getImgIds(catIds=catIds)
img_fileinfo = coco.loadImgs(imgIds)
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

print("Creating Custom {}x{} Dataset".format(new_w, new_h))
resized_bbox_dict = {}
for i in progressbar(range(len(img_fileinfo))):
    # Load image and corresponding annotation
    filename = os.path.join(dataDir, 'images', DataType, img_fileinfo[i]['file_name'])
    img_np = io.imread(filename)
    img_id = img_fileinfo[i]['id']
    annId = coco.getAnnIds(imgIds=[img_id], iscrowd=False) # We don't want bounding boxes that go over groups of objects
    ann = coco.loadAnns(annId)

    if resized_bbox_dict.get(img_id) != None:
        raise Exception("duplicate image IDs in dataset")
    else:
        resized_bbox_dict[img_id] = {cat:[] for cat in catNms}
    
    # Extract all the annotations that are only for the desired object categories
    reduced_ann = []
    for obj in ann:
        if obj['category_id'] in catIds:
            reduced_ann.append(obj)

    # plt.figure(1)
    # plt.imshow(img_np)
    # coco.showAnns(reduced_ann, draw_bbox=True)

    # Resize the image and bounding boxes
    orig_w, orig_h = img_np.shape[1], img_np.shape[0] # Recall width=columns, height=rows (can be confusing)
    w_scale_factor, h_scale_factor = new_w/orig_w, new_h/orig_h

    img_resized = resize(img_np, (new_w, new_h))
    for j in range(len(reduced_ann)):
        x, y, w, h = orig_bbox = reduced_ann[j]['bbox']
        reduced_ann[j]['bbox'] = [x*w_scale_factor, y*h_scale_factor, w*w_scale_factor, h*h_scale_factor]
        # Area Suppression - Only add boxes that have area bigger than area_suppression_factor (pixels^2)
        if (w*w_scale_factor)*(h*h_scale_factor) >= area_suppression_factor:
            resized_bbox_dict[img_id][coco.loadCats(reduced_ann[j]['category_id'])[0]['name']].append(reduced_ann[j]['bbox'])

    # plt.figure(2)
    # fig = plt.imshow(img_resized) # We didn't resize the segmentation, so if that looks bad it's OK, we just care about bounding boxes
    # coco.showAnns(reduced_ann, draw_bbox=True)
    # plt.show()

    # Check that there are still bounding relevant bounding box annotations after area suppression
    isEmpty = True
    for cat in resized_bbox_dict[img_id].keys():
        if resized_bbox_dict[img_id][cat] != []:
            isEmpty = False
            break
    
    if not isEmpty:
        fout = str(img_id) + '.jpg'
        filename = os.path.join(outDir, 'images', DataType, fout)
        io.imsave(filename, img_as_ubyte(img_resized))
    else:
        resized_bbox_dict.pop(img_id)

with open(os.path.join(outDir, 'annotations',"{}bboxes.json".format(DataType)), "w") as outfile:  
    json.dump(resized_bbox_dict, outfile) 
    outfile.close()
    
    '''
    Custom JSON hierarchy:
    img_id
        cat
            bbox1
            bbox2
            bbox3
            ...
        cat 
            bbox1
            bbox2
            bbox3
            ...
    '''

print("Done! Final Dataset size is: {} images".format(len(resized_bbox_dict.keys())))