import IPython
import json
import matplotlib.pyplot as plt
import os
from progressbar import progressbar
import skimage.io as io

import tensorflow as tf
from tensorflow.keras.models import load_model
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



if os.getcwd() != 'D:\Desktop\EE113DB\Custom Model':
    os.chdir('D:\Desktop\EE113DB\Custom Model')

model = load_model('D:\Desktop\EE113DB\Custom Model\yolo-fastestCOCO128-set_inputMODIFIED.h5')
model.summary()

# TODO:
'''
1. Modify network architecture to fit on the H7 (verify with X-CUBE AI) -> (or from scratch???)
2. Generate Custom Training Dataset - only once (1) is set in stone!!!
3. Generate Anchors: 
    https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e
    Need to understand anchors, how to use/modify decode_netout
4. Write Training Script
'''

# NOTE - I can't solidify the network architecture until:
'''
Are we implementing network from scratch? -> Then we can aggressively deallocate -> 200x200? (back of the envelope size calculations)
If we aren't implementing from scratch on H7 -> Have to condense it EVEN FURTHER -> Garbage performance (ask if we don't have to have everything on RAM at once)
One idea is to to make images GrayScale -> 3x reduction in data taken up by input
'''