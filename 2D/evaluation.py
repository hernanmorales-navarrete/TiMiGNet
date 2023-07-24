#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import *

# ACTIVATE GPU USAGE

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

image_path = ''
output_name = ''
model_path = ''

prediction = predict_2D(input_image_path = image_path,
                        output_name = output_path,
                        model_path = model_path,
                        patch_size = 128)
        
prediction = predict_3D(input_image_path = image_path,
                        output_name = output_name,
                        model_path = model_path, 
                        patch_size = 64)
