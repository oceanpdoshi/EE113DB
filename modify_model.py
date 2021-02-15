import json
import os

import tensorflow as tf

curr_model_weights = 'D:/Desktop/EE113DB/Custom Model/old_models/yolo-fastestCOCO128Modified.h5'
new_model_architecture = 'D:/Desktop/EE113DB/Custom Model/yolo-fastestCAR128.json'
new_model_name = 'yolo-fastestCAR128.h5'

# TODO - modify the JSON file to get something that fits (using X-CUBE AI)
with open(new_model_architecture, 'r') as json_file:
    architecture = json.load(json_file)
    model = tf.keras.models.model_from_json(json.dumps(architecture))
    json_file.close()

model.summary()

model.load_weights(curr_model_weights, by_name=True, skip_mismatch=True)
model.save(new_model_name)

# NOTE
'''
- we basically are truncating the feature detector to the 12x12 layer
- Because we changed the number of classes, we are NOT loading pre-trained weights for 
    the object detection block. 
- This should fit using X-CUBE AI!
'''