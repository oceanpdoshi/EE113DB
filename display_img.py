import IPython
import json
import matplotlib.pyplot as plt
import os
from progressbar import progressbar
import skimage.io as io


from bounding_box import draw_bbox

dataDir = 'custom_datasets200'
dataType = 'val2017'

img_filenames = os.listdir(os.path.join(dataDir, 'images', dataType))
img_ids = [os.path.splitext(f)[0] for f in img_filenames] # get rid of '.jpg' extension

with open(os.path.join(dataDir, 'annotations', '{}bboxes.json'.format(dataType) ), 'r') as json_file:
    img_bboxes_json = json.load(json_file)
    json_file.close()

for i in progressbar(range(len(img_ids))):
    id = img_ids[i]
    img_path = os.path.join(dataDir, 'images', dataType, img_filenames[i])
    img_bboxes = img_bboxes_json[id]
    cats = img_bboxes.keys()

    draw_bbox(img_path, img_bboxes, scores=[]) # TODO - fix this later to handle scores from network output