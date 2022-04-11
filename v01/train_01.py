#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.utils import resample
from instancenormalization import InstanceNormalization
from keras.models import load_model
import cv2
import tifffile as tiff
from CycleGAN_model_01 import define_generator, define_discriminator, define_composite_model, train
from datetime import datetime

#%%

###### TENSORFLOW VERSION: 2.7.0
###### KERAS VERSION: 2.7.0


# load all images into a directory into memory

def load_images(path, size = (64, 64)):
    data_list = list()
    for filename in os.listdir(path):
        if not filename.startswith('.'):
        # load and resize images
            pixels = tiff.imread(path + filename)
            pixels = img_to_array(pixels)
            data_list.append(pixels)
  
    return np.asarray(data_list)

path = '/Volumes/Extreme SSD/CycleGANs/images/' # path to main folder (containing subfolders img0 & img1)
path_img0 = 'patches_64_img0_prueba/'
path_img1 = 'patches_64_img1_prueba/'

# LOAD DATA A (IMG 0: TARGET)

dataA_all = load_images(path + path_img0) # TARGET
print('Loaded data A: ', dataA_all.shape)

dataA = resample(dataA_all, replace = False, n_samples = 10, random_state = 27)

# LOAD DATA B (IMG 1: ORIGIN)

dataB_all = load_images(path + path_img1) # CAMBIA
print('Loaded data B: ', dataB_all.shape) 

dataB = resample(dataB_all, replace = False, n_samples = 10, random_state = 27)


#%% load image data

data = [dataA, dataB]
#data[0] = np.reshape(data[0], (10, 64, 64, 1))
#data[1] = np.reshape(data[1], (10, 64, 64, 1))
print('Loaded', data[0].shape, data[1].shape)

#%%

# preprocess data to values [-1, 1] (tanh activation)

def preprocess_data_norm(data):
    X1, X2 = data[0], data[1]
    X1 = cv2.normalize(X1, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    X2 = cv2.normalize(X2, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return [X1, X2]

dataset = preprocess_data_norm(data)
#plt.imshow(dataset[0][0])
#%%


# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]

# define generators, discriminator and composite model

# generator A -> B
g_model_AtoB = define_generator(image_shape)
# generator B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite model A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite model B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)



#%% 

# train models (MODIFICAR MANUALMENTE CANTIDAD DE EPOCHS)

start1 = datetime.now()

train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs = 1)

stop1 = datetime.now()

execution_time = stop1 - start1

print('execution time is: ', execution_time)


#### Aquí termina la sesión de entrenamiento ###
# se debe obtener como resultado modelos para ambos generadores, en formato .h5 en la carpeta de trabajo
# además, se guarda una figura que muestra ejemplos de las imagenes generadas por ambos generadores

# recordar que A: img 0 (TARGET) y B: img 1 (ORIGEN)


#%%
############################################### EVALUATION #####################################

# select a random sample of images from the dataset

def select_sample(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples) # choose random instances
    X = dataset[ix] # retrieve selected images
    return X


# plot the image, its translation and the reconstruction

def show_plot(imagesX, imagesY1, imagesY2):
    images = np.vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Translated', 'Reconstructed']
    # scale from [-1, 1] to [0, 1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        plt.subplot(1, len(images), 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])
    plt.show()
    
    
# load dataset

A_data = resample(dataA_all,
                  replace = True, # if n_samples < 50
                  n_samples = 50,
                  random_state = 27)

B_data = resample(dataB_all,
                  replace = True,
                  n_samples = 50,
                  random_state = 27)

# scale the data [-1, 1]

A_data = cv2.normalize(A_data, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
B_data = cv2.normalize(B_data, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)


# load the models

cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('/Volumes/Extreme SSD/CycleGANs/scripts/g_model_AtoB_000010.h5', cust)
model_BtoA = load_model('/Volumes/Extreme SSD/CycleGANs/scripts/g_model_BtoA_000010.h5', cust)

#%% plot A -> B -> A (TARGET TO ORIGIN TO TARGET)

A_real = select_sample(A_data, 1) # selecciona una imagen (img 0)
B_generated = model_AtoB.predict(A_real) # genera imagen (img 1)
A_reconstructed = model_BtoA.predict(B_generated) # reconstruye img 0 a partir de img 1 generada
show_plot(A_real, B_generated, A_reconstructed) # muestra el resultado

#%%
# plot B -> A -> B (ORIGIN TO TARGET TO ORIGIN) --- ***** ESTA NOS INTERESA (va de img 1 a img 0)

B_real = select_sample(B_data, 1) # selecciona una imagen (img 1)
A_generated = model_BtoA.predict(B_real) # genera una imagen (img 0)
B_reconstructed = model_AtoB.predict(A_generated) # reconstruye img 1 a partir de img 0 generada
show_plot(B_real, A_generated, B_reconstructed) # muestra el resultado

#%%
# #%% LOAD A SINGLE CUSTOM IMAGE

# test_image_path = '/Volumes/Extreme SSD/CycleGANs/images/patches_64_img1/image_0_119.tif'
# test_image = tiff.imread(test_image_path)
# test_image = img_to_array(test_image)
# test_image_input = np.array([test_image])
# test_image_input = cv2.normalize(test_image_input, None, -1, 1, cv2.NORM_MINMAX, dtype = cv2.CV_32F)


# img_generated = model_BtoA.predict(test_image_input)
# img_reconstructed = model_BtoA.predict(img_generated)
# show_plot(test_image_input, img_generated, img_reconstructed)





