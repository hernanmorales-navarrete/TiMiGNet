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
        

# =============================================================================
#  TRAINING PROCESS - 2D MODEL
# =============================================================================

# GENERATE DATASET

a_0 = ''
a_1 = ''

b_0 = ''
b_1 = ''

images_0 = [a_0, b_0]
images_1 = [a_1, b_1]

dataset_0 = generate_dataset(model_type = '2d',
                              images = images_0,
                              patch_shape = (128, 128),
                              patch_step = 128,
                              n_samples = 2500,
                              seed_a = 27,
                              seed_b = 57,
                              shuffle_ = True)

dataset_1 = generate_dataset(model_type = '2d',
                              images = images_1,
                              patch_shape = (128, 128),
                              patch_step = 128,
                              n_samples = 2500,
                              seed_a = 27,
                              seed_b = 57,
                              shuffle_ = True)

dataset = merge_datasets(datasets = [dataset_0, dataset_1],
                          seed_a = 27,
                          seed_b = 57,
                          shuffle_ = True)

# train the model

training = train_model_2D(dataset = dataset,
                          batch_size = 1, 
                          epochs = 100, 
                          save_freq = 10) 

