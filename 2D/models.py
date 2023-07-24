#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import all necessary packages and libraries

import random
import numpy as np
import tensorflow as tf
#from keras.models import Input, Model # for tensorflow 2.10 or newer
from tensorflow.keras import Input, Model
from keras.models import load_model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from keras.layers import Conv3D, Conv3DTranspose
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
from instancenormalization import InstanceNormalization # (script adjunto)
import tifffile as tiff
import pandas as pd
import cv2
from patchify import patchify
from sklearn.utils import shuffle, resample
from datetime import datetime
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from empatches import EMPatches
import os
from scipy.ndimage import maximum_filter


# =============================================================================
# 2D MODEL
# =============================================================================

# ------------ DISCRIMINATOR --------------- 

def define_discriminator_2D(image_shape):
    init = RandomNormal(stddev = 0.02, seed = 0) # weight initialization
    in_image = Input(shape = image_shape)
    
    #C64: 4x4 Kernel - Stride 2x2
    d = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(in_image)
    d = LeakyReLU(alpha = 0.2)(d)
    #C128: 4x4 Kernel - Stride 2x2
    d = Conv2D(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #C256: 4x4 Kernel - Stride 2x2
    d = Conv2D(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #C512: 4x4 Kern el - Stride 2x2
    d = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2,2), padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #OUTPUT - No stride
    output = Conv2D(filters = 1, kernel_size = (4, 4), padding = 'same', kernel_initializer = init)(d)
    
    #Define Model
    model = Model(in_image, output)
    
    #Compile Model
    model.compile(loss = 'mse', 
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5),
                  loss_weights = [0.5])
    
    return model

# ---------------- GENERATOR ----------------

# --- RESNET BLOCK ---

def resnet_block_2D(n_filters, input_layer):
    
    init = RandomNormal(stddev = 0.02, seed = 0)   
    g = Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = init)(input_layer)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    g = Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Concatenate()([g, input_layer])
    
    return g


def define_generator_2D(image_shape, rn_blocks = 6): # 6 resnet blocks for 128 dim or lower. 9 for 256 or higher. (original paper)
    init = RandomNormal(stddev = 0.02, seed = 0)
    in_image = Input(shape = image_shape)
    
    # #C7S1-64  -> kernel_size = 7, stride = 1, filters = 64
    g = Conv2D(filters = 64, kernel_size = (7, 7), strides = (1, 1), padding = 'same', kernel_initializer = init)(in_image)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #dk  -> 3x3 Conv-InstNorm-ReLU, k filtros, stride = 2
    # #d128
    g = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)

    # #d256
    g = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    #R256 (6 resnet blocks)
    for block_ in range(rn_blocks):
        g = resnet_block_2D(256, g)
    
    #uk -> 3x3 TransposedConv-InstanceNorm-ReLU, k filtros, stride = 2
    #u128 
    g = Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #u64
    g = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #C7S1-3 # number of filters is 1 because there is only 1 channel (gray)
    g = Conv2D(filters = 1, kernel_size = (7, 7), strides = (1, 1), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    output_image = Activation('tanh')(g) # tanh for values between [-1, 1]
    
    model = Model(inputs = in_image, outputs = output_image)
    return model

    
# ------------------ COMPOSITE MODEL -----------------


def define_composite_model_2D(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True # 1st generator: trainable
    d_model.trainable = False # discriminator: non-trainable
    g_model_2.trainable = False # 2nd generator : non-trainable
    
    # LOSSES
    
    # - - - adversarial loss - - -
    input_gen = Input(shape = image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    
    # # - - - identity loss - - - 
    input_id = Input(shape = image_shape)
    output_id = g_model_1(input_id)
    
    # # - - - cycle loss - forward - - - 
    output_f = g_model_2(gen1_out)

    # # - - - cycle loss - backwards - - - 
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    
    # # define model
    model = Model(inputs = [input_gen, input_id], outputs = [output_d, output_id, output_f, output_b])
    
    # compile model
    model.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5),
                  loss = ['mse', 'mae', 'mae', 'mae'],
                  loss_weights = [1, 5, 10, 10])
    
    return model

# select a batch of random samples, returns images and target
def generate_real_samples_2D(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1)) # generate "real" class labels (1)
    return X, y

# generate a batch of (fake) images, returns images and targets
def generate_fake_samples_2D(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1)) # create "fake" class labels (0)
    return X, y

# saves the generator model to files 
def save_models_2D(step, g_model_AtoB, g_model_BtoA,output_dir):
    filename1 = output_dir+'/models/g_model_AtoB_2D_%06d.h5' % (step + 1) # save the first generator model
    g_model_AtoB.save(filename1)
    
    filename2 = output_dir+'/models/g_model_BtoA_2D_%06d.h5' % (step + 1)
    g_model_BtoA.save(filename2)
    
    print('>Saved: %s and %s' % (filename1, filename2))
    
    
# model prediction on 5 random sample images
def summarize_performance_2D(step, g_model, trainX, name, output_dir, n_samples = 5):
    
    X_in, _ = generate_real_samples_2D(trainX, n_samples, 0) # select a sample of input images
    X_out, _ = generate_fake_samples_2D(g_model, X_in, 0)
    X_in = (X_in + 1) / 2.0 # scale all pixels from [-1, 1] to [0, 1]
    X_out = (X_out + 1) / 2.0
    
    
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i])
    
    # plot translated images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i])
    
    # save plot to file
    filename1 = output_dir+'/example_images/%s_2D_generated_plot_%06d.png' % (name, (step + 1))
    plt.savefig(filename1, dpi = 1200)
    plt.close()
    
    
def update_image_pool_2D(pool, images, max_size = 50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5: 
            selected.append(image) 
        else: 
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)

    
# train cycleGAN models

def train_2D(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, output_dir, batch_size = 1, save_freq = 10, epochs = 1):
    # properties of training run
    n_epochs, n_batch = epochs, batch_size 
    
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    
    # unpack dataset
    trainA = dataset[0]
    trainB = dataset[1]

    # prepare image pool for fake images
    poolA, poolB = list(), list()
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    #SAVE LOSSES
    
    list_dA_loss1, list_dA_loss2, list_dB_loss1, list_dB_loss2, list_g_loss1, list_g_loss2 = [], [], [], [], [], []
    
    epoch_count = []
    epo_counter = 0.20
    
    idx = 0
    idx_list = []
    
    
    # directory for sample images
    os.mkdir(output_dir+'/example_images/')
    
    # directory for models
    os.mkdir(output_dir+'/models/')
    
    
    try: 
        for i in range(n_steps):
            
            # select a batch of real samples from each domain (A and B)
            
            X_realA = np.expand_dims(trainA[idx], axis = 0)
            y_realA = np.ones((1, n_patch, n_patch, 1))
            X_realB = np.expand_dims(trainB[idx], axis = 0)
            y_realB = np.ones((1, n_patch, n_patch, 1))
            
            # generate a batch of fake samples using both B to A and A to B generators
            
            X_fakeA = g_model_BtoA.predict(X_realB)
            y_fakeA = np.zeros((len(X_fakeA), n_patch, n_patch, 1))
            
            X_fakeB = g_model_AtoB.predict(X_realA)
            y_fakeB = np.zeros((len(X_fakeB), n_patch, n_patch, 1))
                
            # update fake images in the pool
            X_fakeA = update_image_pool_2D(poolA, X_fakeA)
            X_fakeB = update_image_pool_2D(poolB, X_fakeB)
                
            # update generator A to B via the composite model
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
                
            # update generator B to A via the composite model
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            
            # summarize perfomance
            print('Iteration>%d/%d, dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' % (i + 1, n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
            
            # iteration counter
            idx_list.append(idx)
            idx += 1
            
            
            # summarize performance and save models every "save_freq" epochs
            
            if (i + 1) % (bat_per_epo * save_freq) == 0:
                # plot A -> B translation
                summarize_performance_2D(i, g_model_AtoB, trainA, 'AtoB', output_dir)
                # plot B -> A translation
                summarize_performance_2D(i, g_model_BtoA, trainB, 'BtoA', output_dir)
                # save the models                
                save_models_2D(i, g_model_AtoB, g_model_BtoA, output_dir)
                    
            # save five losses per epoch
            if (i + 1) % (bat_per_epo*0.20) == 0:
                    
                epoch_count.append((np.ceil(i / bat_per_epo) -1 ) + epo_counter)
                list_dA_loss1.append(dA_loss1)
                list_dA_loss2.append(dA_loss2)
                list_dB_loss1.append(dB_loss1)
                list_dB_loss2.append(dB_loss2)
                list_g_loss1.append(g_loss1)
                list_g_loss2.append(g_loss2)
                epo_counter += 0.20
            
  
            # reset iteration counter once it's matched the total number of images
            if (idx) % (bat_per_epo) == 0: 
                idx = 0
            
              # reset counter for saving losses
            if (i + 1) % (bat_per_epo) == 0: 
                epo_counter = 0.20
                
            
        df_losses = pd.DataFrame({'epoch': epoch_count,
                                          'dA_loss1': list_dA_loss1,
                                          'dA_loss2': list_dA_loss2,
                                          'dB_loss1': list_dB_loss1,
                                          'dB_loss2': list_dB_loss2,
                                          'g_loss1': list_g_loss1,
                                          'g_loss2': list_g_loss2})
        
        df_losses.to_csv(output_dir+'/LOSSES_TRAINING.csv')
        
        return df_losses

    except:
        
        df_losses = pd.DataFrame({'epoch': epoch_count,
                                          'dA_loss1': list_dA_loss1,
                                          'dA_loss2': list_dA_loss2,
                                          'dB_loss1': list_dB_loss1,
                                          'dB_loss2': list_dB_loss2,
                                          'g_loss1': list_g_loss1,
                                          'g_loss2': list_g_loss2})
        
        
        df_losses.to_csv(output_dir+'/LOSSES_TRAINING.CSV')
        
        return df_losses


# =============================================================================
# 3D MODEL
# =============================================================================


# ------------ DISCRIMINATOR --------------- 

def define_discriminator_3D(image_shape):
    init = RandomNormal(stddev = 0.02, seed = 0) # weight initialization
    in_image = Input(shape = image_shape)
    
    #C64: 4x4 Kernel - Stride 2x2
    d = Conv3D(filters = 64, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = init)(in_image)
    d = LeakyReLU(alpha = 0.2)(d)
    #C128: 4x4 Kernel - Stride 2x2
    d = Conv3D(filters = 128, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #C256: 4x4 Kernel - Stride 2x2
    d = Conv3D(filters = 256, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #C512: 4x4 Kern el - Stride 2x2
    d = Conv3D(filters = 512, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    #OUTPUT - No stride
    output = Conv3D(filters = 1, kernel_size = 4, padding = 'same', kernel_initializer = init)(d)
    
    #Define Model
    model = Model(in_image, output)
    
    #Compile Model
    model.compile(loss = 'mse', 
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5),
                  loss_weights = [0.5])
    
    return model

# ---------------- GENERATOR ----------------

# --- RESNET BLOCK ---

def resnet_block_3D(n_filters, input_layer):
    
    init = RandomNormal(stddev = 0.02, seed = 0)   
    g = Conv3D(filters = n_filters, kernel_size = 3,  padding = 'same', kernel_initializer = init)(input_layer)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    g = Conv3D(filters = n_filters, kernel_size = 3, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Concatenate()([g, input_layer])
    
    return g


# --- GENERATOR ---

def define_generator_3D(image_shape, rn_blocks = 6):
    init = RandomNormal(stddev = 0.02, seed = 0)
    in_image = Input(shape = image_shape)
    
    # #C7S1-64  -> kernel_size = 7, stride = 1, filters = 64
    g = Conv3D(filters = 64, kernel_size = 7, strides = 1, padding = 'same', kernel_initializer = init)(in_image)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #dk  -> 3x3 Conv-InstNorm-ReLU, k filters, stride = 2
    # #d128
    g = Conv3D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)

    # #d256
    g = Conv3D(filters = 256, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    #R256 (6 resnet blocks)
    for block_ in range(rn_blocks):
        g = resnet_block_3D(256, g)
    
    #uk -> 3x3 TransposedConv-InstanceNorm-ReLU, k filtros, stride = 2
    #u128 
    g = Conv3DTranspose(filters = 128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #u64
    g = Conv3DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #C7S1-3
    g = Conv3D(filters = 1, kernel_size = 7, strides = 1, padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    output_image = Activation('tanh')(g) # tanh for values between [-1, 1]
    
    model = Model(inputs = in_image, outputs = output_image)
    return model

    
# ------------------ COMPOSITE MODEL -----------------


def define_composite_model_3D(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True # 1er generator: trainable
    d_model.trainable = False # discriminator: non-trainable
    g_model_2.trainable = False # 2do generator : non-trainable
    
    # model losses:
    
    # - - - adversarial loss - - -
    input_gen = Input(shape = image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    
    # # - - - identity loss - - - 
    input_id = Input(shape = image_shape)
    output_id = g_model_1(input_id)
    
    # # - - - cycle loss - forward - - - 
    output_f = g_model_2(gen1_out)

    # # - - - cycle loss - backwards - - - 
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    
    # # define model
    model = Model(inputs = [input_gen, input_id], outputs = [output_d, output_id, output_f, output_b])
    
    # compile model
    model.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5),
                  loss = ['mse', 'mae', 'mae', 'mae'],
                  loss_weights = [1, 5, 10, 10])
    
    return model


# select a batch of random samples, returns images and target
def generate_real_samples_3D(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, patch_shape, 1)) # generate "real" class labels (1)
    return X, y

# generate a batch of (fake) images, returns images and targets
def generate_fake_samples_3D(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, patch_shape, 1)) # create "fake" class labels (0)
    return X, y

# saves the generator model to files 
def save_models_3D(step, g_model_AtoB, g_model_BtoA,output_dir):
    filename1 = output_dir+'/models/g_model_AtoB_3D_%06d.h5' % (step + 1) # save the first generator model
    g_model_AtoB.save(filename1)
    
    filename2 = output_dir+'/models/g_model_BtoA_3D_%06d.h5' % (step + 1)
    g_model_BtoA.save(filename2)
    
    print('>Saved: %s and %s' % (filename1, filename2))
      
    
def summarize_performance_3D(step, g_model, trainX, name, output_dir, n_samples = 1):
    
    X_in, _ = generate_real_samples_3D(trainX, n_samples, 0) # select a sample of input images
    X_out, _ = generate_fake_samples_3D(g_model, X_in, 0)
    X_in = (X_in + 1) / 2.0 # scale all pixels from [-1, 1] to [0, 1]
    X_out = (X_out + 1) / 2.0

    tiff.imwrite(output_dir+'/example_images/' + name + '_3D_original_' + str(step + 1) + '.tif', X_in)
    tiff.imwrite(output_dir+'/example_images/' + name + '_3D_generated_' + str(step + 1) + '.tif', X_out)
    
    
def update_image_pool_3D(pool, images, max_size = 50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            selected.append(image)
        else: 
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


# train 3D CycleGAN model

def train_3D(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, output_dir, batch_size = 1, save_freq = 10, epochs = 1):
    
    n_epochs, n_batch = epochs, batch_size # recommended in paper - batch = 1
    
    # determine the output shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    
    # unpack dataset
    trainA = dataset[0]
    trainB = dataset[1]

    # prepare image pool for fake images
    poolA, poolB = list(), list()
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    #SAVE LOSSES
    
    # training losses
    list_dA_loss1, list_dA_loss2, list_dB_loss1, list_dB_loss2, list_g_loss1, list_g_loss2 = [], [], [], [], [], []

    
    epoch_count = []
    epo_counter = 0.20

    idx = 0
    idx_list = []
    
    # directory for sample images
    os.mkdir(output_dir+'/example_images/')
    
    # directory for models
    os.mkdir(output_dir+'/models/')

    try: 
        for i in range(n_steps):
            # select a batch of real samples from each domain (A and B)
            
            X_realA = np.expand_dims(trainA[idx], axis = 0)
            y_realA = np.ones((1, n_patch, n_patch, n_patch, 1))
            X_realB = np.expand_dims(trainB[idx], axis = 0)
            y_realB = np.ones((1, n_patch, n_patch, n_patch, 1))
                
            # generate a batch of fake samples using both B to A and A to B generators
            
            X_fakeA = g_model_BtoA.predict(X_realB)
            y_fakeA = np.zeros((len(X_fakeA), n_patch, n_patch, n_patch, 1))
            
            X_fakeB = g_model_AtoB.predict(X_realA)
            y_fakeB = np.zeros((len(X_fakeB), n_patch, n_patch, n_patch, 1))

            # update fake images in the pool. (buffer of 50 images - paper)
            X_fakeA = update_image_pool_2D(poolA, X_fakeA)
            X_fakeB = update_image_pool_2D(poolB, X_fakeB)
                
            # update generator A to B via the composite model
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
                
            # update generator B to A via the composite model
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            
            
    
            # summarize perfomance
            print('Iteration > %d/%d: dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' % (i + 1, n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
            
            # iteration counter
            
            idx_list.append(idx)
            idx += 1 
            
            
            # summarize performance and save models every "save_freq" epochs
            
            if (i + 1) % (bat_per_epo * save_freq) == 0:
                # plot A -> B translation
                summarize_performance_3D(i, g_model_AtoB, trainA, 'AtoB', output_dir)
                # plot B -> A translation
                summarize_performance_3D(i, g_model_BtoA, trainB, 'BtoA', output_dir)
                # save the models                
                save_models_3D(i, g_model_AtoB, g_model_BtoA,output_dir)
                    
            # save training losses. 5 per epoch.
            if (i + 1) % (bat_per_epo*0.20) == 0:
                    
                epoch_count.append((np.ceil(i / bat_per_epo) -1 ) + epo_counter)
                list_dA_loss1.append(dA_loss1)
                list_dA_loss2.append(dA_loss2)
                list_dB_loss1.append(dB_loss1)
                list_dB_loss2.append(dB_loss2)
                list_g_loss1.append(g_loss1)
                list_g_loss2.append(g_loss2)
                epo_counter += 0.20
                
            # reset iteration counter
            if (idx) % (bat_per_epo) == 0: 
                idx = 0
                
                
              # reset counter for saving losses.
              
            if (i + 1) % (bat_per_epo) == 0: 
                epo_counter = 0.20
                
            
        df_losses = pd.DataFrame({'epoch': epoch_count,
                                          'dA_loss1': list_dA_loss1,
                                          'dA_loss2': list_dA_loss2,
                                          'dB_loss1': list_dB_loss1,
                                          'dB_loss2': list_dB_loss2,
                                          'g_loss1': list_g_loss1,
                                          'g_loss2': list_g_loss2})
        
        df_losses.to_csv(output_dir+'/LOSSES_TRAINING.csv')
        
        return df_losses

    except:
        
        df_losses = pd.DataFrame({'epoch': epoch_count,
                                          'dA_loss1': list_dA_loss1,
                                          'dA_loss2': list_dA_loss2,
                                          'dB_loss1': list_dB_loss1,
                                          'dB_loss2': list_dB_loss2,
                                          'g_loss1': list_g_loss1,
                                          'g_loss2': list_g_loss2})
       

        df_losses.to_csv(output_dir+'/LOSSES_TRAINING.CSV')
        return df_losses



# =============================================================================
# PATCH GENERATORS (2D & 3D)
# =============================================================================

def normalize_image(img, percentiles):

	
    img_min = np.amin(img)
    img_max = np.amax(img)
    imgArray = img.flatten()
    # remove backgoround
    low_thres = np.percentile(imgArray,percentiles[0])
    clipped_imgArray = imgArray[imgArray > low_thres]
    # remove upper outliers 
    high_thres = np.percentile(clipped_imgArray,percentiles[1])
    # Ste the values in the image
    img[img > high_thres] = high_thres
    img = img - low_thres
    img[img < low_thres] = 0	
	
    newimg_min = np.amin(img)
    newimg_max = np.amax(img)	
    
    print('Intensity from (%d , %d) to  (%d, %d) ' % (img_min, img_max, newimg_min, newimg_max), '\n')
    
    return img


def patches_2d(original_img, percentiles, patch_shape, patch_step):
       
    img_patches = []
    img = tiff.imread(original_img)
    
    if percentiles[0] > 0 or percentiles[1] < 100:
       img = normalize_image(img, percentiles)
    
    img = np.squeeze(img)
    img_shape = img.shape
    img = np.reshape(img, -1)
    img = cv2.normalize(img, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = np.reshape(img, img_shape)

    for image in range(img.shape[0]):
        
        full_image = img[image]
        patches_image = patchify(full_image, patch_size = patch_shape, step = patch_step)
        
        for i in range(patches_image.shape[0]):
            for j in range(patches_image.shape[1]):
                
                patch = patches_image[i, j, :, :]
                patch = np.expand_dims(patch, -1)
                img_patches.append(patch)
                
    return np.array(img_patches)

def patches_3d(original_img, percentiles, patch_shape, patch_step):
    
    img_patches = []
    img = tiff.imread(original_img)
    
    if percentiles[0] > 0 or percentiles[1] < 100:
       img = normalize_image(img, percentiles)
       
    img_shape = img.shape
    img = np.reshape(img, -1)
    img = cv2.normalize(img, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = np.reshape(img, img_shape)
    patches = patchify(img, patch_size = patch_shape, step = patch_step)
    patches = np.reshape(patches, (-1, patch_shape[0], patch_shape[1], patch_shape[2]))
    
    for patch in patches:   	 
        img_patches.append(np.expand_dims(patch, -1))
    
    return np.array(img_patches)



def filter_patches(img_patches, threshold):
    
    # Filter patch with max intensity higher than a threshold 
    filtered_patches = []
    thres = (2.0 * threshold) - 1.0
    imgA = img_patches[0]
    imgB = img_patches[1]
    dataSet_size = imgA.shape
    Npatches = dataSet_size[0]   
    
    newimgA = []
    newimgB = []
    
    for idx in range(Npatches):
    	if np.mean(imgB[idx,:,:,:]) > thres:
    	   newimgA.append(imgA[idx,:,:,:]) 
    	   newimgB.append(imgB[idx,:,:,:]) 
    	   #print('thres : '+ str(np.amax(imgB[idx,:,:,:])))
    	
    newimgA = np.array(newimgA)
    newimgB = np.array(newimgB)
	
    filtered_patches.append(newimgA)
    filtered_patches.append(newimgB)
   
    #print(thres)
    #print(len(filtered_patches))
    #print(filtered_patches[0].shape)

    return filtered_patches





# =============================================================================
# MAIN FUNCTIONS TO TRAIN 2D & 3D CYCLEGAN MODELS
# =============================================================================

def train_model_2D(dataset, batch_size, epochs, save_freq, output_dir):
    
    image_shape = dataset[0].shape[1:]

    g_model_AtoB = define_generator_2D(image_shape)
    g_model_BtoA = define_generator_2D(image_shape)
    d_model_A = define_discriminator_2D(image_shape)
    d_model_B = define_discriminator_2D(image_shape)
    c_model_AtoB = define_composite_model_2D(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    c_model_BtoA = define_composite_model_2D(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    
    
    start1 = datetime.now()
    
    train = train_2D(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, output_dir, batch_size = batch_size, save_freq = save_freq, epochs = epochs)
    
    stop1 = datetime.now()
    
    execution_time = stop1 - start1
    
    print('execution time is: ', execution_time)    
    
    results = train
    
    return results
    
def train_model_3D(dataset, batch_size, epochs, save_freq, output_dir):
    
    image_shape = dataset[0].shape[1:]

    g_model_AtoB = define_generator_3D(image_shape)
    g_model_BtoA = define_generator_3D(image_shape)
    d_model_A = define_discriminator_3D(image_shape)
    d_model_B = define_discriminator_3D(image_shape)
    c_model_AtoB = define_composite_model_3D(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    c_model_BtoA = define_composite_model_3D(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    
    
    start1 = datetime.now()
    
    train = train_3D(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, output_dir, batch_size = batch_size, save_freq = save_freq, epochs = epochs)
    
    stop1 = datetime.now()
    
    execution_time = stop1 - start1
    
    print('execution time is: ', execution_time)    
    
    results = train
    
    return results



# =============================================================================
#  DATASET GENERATOR FUNCTIONS.
# =============================================================================

def generate_datasets(model_type, images_folder_source, images_folder_target, images_extension, patch_shape, patch_step, threshold, percentiles, n_samples, seed_a, seed_b, shuffle_ = True):

    data_a = []
    data_b = []
        
    if model_type == '2d':
    
    	for filename in os.listdir(images_folder_source):
    		if filename.endswith(images_extension): 
    			image = os.path.join(images_folder_source, filename)
    			print('Importing image A: ', image)
    			patches = patches_2d(image, percentiles, patch_shape, patch_step)
    			data_a.append(patches)
	    	else:
		     continue

    	for filename in os.listdir(images_folder_target):
    		if filename.endswith(images_extension): 
    			image = os.path.join(images_folder_target, filename)
    			print('Importing image B: ', image)
    			patches = patches_2d(image, percentiles, patch_shape, patch_step)
    			data_b.append(patches)
	    	else:
		     continue
		
    
    elif model_type == '3d':
 
     	for filename in os.listdir(images_folder_source):
     		if filename.endswith(images_extension):
     			image = os.path.join(images_folder_source, filename)
     			print('Importing image A: ', image)
     			patches = patches_3d(original_img = image, percentiles=percentiles, patch_shape = patch_shape, patch_step = patch_step)
     			data_a.append(patches)
     		else:
     			continue

     	for filename in os.listdir(images_folder_target):
     		if filename.endswith(images_extension):
     			image = os.path.join(images_folder_target, filename)
     			print('Importing image B: ', image)
     			patches = patches_3d(original_img = image, percentiles=percentiles, patch_shape = patch_shape, patch_step = patch_step)
     			data_b.append(patches)
     		else:
     			continue
		        
    
    data = [np.concatenate(data_a), np.concatenate(data_b)]
    print('total number of 3D-patches for image A: ', len(data[0]))
    print('total number of 3D-patches for image B: ', len(data[1]), '\n')
        
    if threshold > 0:
    	data = filter_patches(data, threshold)
    	print('total number of patches for image A after filtering: ', len(data[0]))
    	print('total number of patches for image B after filtering: ', len(data[1]), '\n')
        
    if shuffle_:
    
        data[0] = shuffle(data[0], n_samples = n_samples, random_state = seed_a)
        data[1] = shuffle(data[1], n_samples = n_samples, random_state = seed_b)
                
        print('%d %s-patches were loaded for image A - SHUFFLED' % (len(data[0]), str(model_type).upper()))
        print('%d %s-patches were loaded for image B - SHUFFLED' % (len(data[1]), str(model_type).upper()), '\n')
        
        return data
            
    else:
       	if n_samples > len(data[0]):
       	   n_samples = len(data[0])
    	   
        data[0] = resample(data[0], n_samples = n_samples, random_state = 0)
        data[1] = resample(data[1], n_samples = n_samples, random_state = 0)
        
        print('%d %s-patches were loaded for image A - NON-SHUFFLED' % (len(data[0]), str(model_type).upper()))
        print('%d %s-patches were loaded for image B - NON-SHUFFLED' % (len(data[1]), str(model_type).upper()), '\n')
                
        return data
    
# =============================================================================
#     
# # PREDICTION/EVALUATION FUNCTIONS
# 
# =============================================================================


# PRE AND POST PROCESSING FUNCTIONS

def image_preprocessing(image, percentiles, min_val, max_val):
    
    img = tiff.imread(image)
    
    if percentiles[0] > 0 or percentiles[1] < 100:
    	img = normalize_image(img, percentiles)
    	
    img_shape = img.shape
    img = np.reshape(img, newshape = -1)
    img = cv2.normalize(img, None, alpha = min_val, beta = max_val, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = np.reshape(img, newshape = img_shape)
    
    return img
    
def image_postprocessing_norm(prediction, percentiles):

    prediction = cv2.normalize(prediction, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)

    if percentiles[0] > 0 or percentiles[1] < 100:
    	prediction = normalize_image(prediction, percentiles)
    	    
    to_uint16 = cv2.normalize(prediction, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
    
    return to_uint16


# MODEL PREDICTION FUNCTIONS

def translate_smooth(image, model, window_size, subdivisions = 2):
    
    start1 = datetime.now()
    
    image_shape = image.shape
    npad = ((0, 0), (window_size, window_size), (window_size, window_size))
    image = np.pad(image, pad_width = npad, mode = 'mean')
    
    translated = []
    n_img = 1
    
    for img in image:

        if (100 * n_img / image_shape[0]) % (10) == 0:
        	print('\nimage %d of %d' % (n_img, image_shape[0]))
        img = np.expand_dims(img, 2)
        
        predictions_smooth = predict_img_with_smooth_windowing(img,
                                                               window_size = window_size,
                                                               subdivisions = subdivisions,
                                                               nb_classes = 1,
                                                               pred_func=(lambda beta: model.predict((beta), verbose=0, use_multiprocessing=True)))
                                                                
        translated.append(predictions_smooth)
        n_img += 1
        
    translated_np = np.squeeze(np.array(translated))
    translated_np = translated_np[:, window_size:image_shape[1] + window_size, window_size:image_shape[2] + window_size]
    
        
    stop1 = datetime.now()   
    execution_time = stop1 - start1 
    print('execution time is: ', execution_time)  
    
    return translated_np



def translate_3D_EMP(image, model, patch_size, overlap = 0.0, mode = 'avg'):
    
    start1 = datetime.now()

    patch_shape = (patch_size, patch_size, patch_size)

    dim_0 = int(((np.ceil(image.shape[0] / patch_shape[0]) * patch_shape[0]) - image.shape[0])/2)
    dim_1 = int(((np.ceil(image.shape[1] / patch_shape[0]) * patch_shape[0]) - image.shape[1])/2)
    dim_2 = int(((np.ceil(image.shape[2] / patch_shape[0]) * patch_shape[0]) - image.shape[2])/2)
    
    npad = ((dim_0, dim_0), (dim_1, dim_1), (dim_2, dim_2))
    img = np.pad(image, pad_width = npad, mode = 'mean')
    
    patch_counter = 1
    emp = EMPatches()
    
    prediction = []
    prediction_patches = []
    
    patches, indices  = emp.extract_patches(img, patchsize=patch_size, overlap=overlap, vox=True)
    
    for patch in patches:
        
        if (100 * patch_counter / len(patches)) % (10) == 0:
        	print('patch %d of %d' % (patch_counter, len(patches)))
        	
        patch = np.expand_dims(patch, 0)
        patch = model.predict(patch, verbose=0, use_multiprocessing=True)
        patch = np.squeeze(patch)
        prediction_patches.append(patch)
        patch_counter += 1
        
    prediction.append(emp.merge_patches(prediction_patches, indices, mode = mode))
    prediction = np.squeeze(np.array(prediction))
    prediction = prediction[dim_0:image.shape[0] + dim_0, dim_1:image.shape[1] + dim_1, dim_2:image.shape[2] + dim_2]
    
        
    stop1 = datetime.now()
    
    execution_time = stop1 - start1
    
    print('execution time is: ', execution_time)  
    
    return prediction




def predict_2D(input_image_path, percentiles, output_name, model_path, patch_size):
    
    instance_norm = {'InstanceNormalization': InstanceNormalization}
    model = load_model(model_path, instance_norm)
        	   
    pred = image_preprocessing(image = input_image_path,
                               percentiles = percentiles,
                               min_val = -1,                               
                               max_val = 1)
    
    pred = translate_smooth(image = pred,
                            model = model,
                            window_size = patch_size,
                            subdivisions = 2)
    
    pred = image_postprocessing_norm(pred, percentiles)
    
    tiff.imwrite(output_name, pred)
    
    return pred
    

def predict_3D(input_image_path, percentiles, output_name, model_path, patch_size, overlap_pct = 0.125):
    
    instance_norm = {'InstanceNormalization': InstanceNormalization}
    model = load_model(model_path, instance_norm)
       
    pred = image_preprocessing(image = input_image_path,
                               percentiles = percentiles,    
                               min_val = -1,
                               max_val = 1)

    pred = translate_3D_EMP(image = pred,
                            model = model,
                            patch_size = patch_size,
                            overlap = overlap_pct,
                            mode = 'avg')
    
    pred = image_postprocessing_norm(pred, percentiles)
    
    tiff.imwrite(output_name, pred)
    
    return pred
    

