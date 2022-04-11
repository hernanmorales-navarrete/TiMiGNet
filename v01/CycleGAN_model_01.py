#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization, ReLU
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
from instancenormalization import InstanceNormalization # (script adjunto)


###### TENSORFLOW VERSION: 2.7.0
###### KERAS VERSION: 2.7.0



# ------------ DISCRIMINATOR --------------- 

def define_discriminator(image_shape):
    init = RandomNormal(stddev = 0.02) # weight initialization
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

def resnet_block(n_filters, input_layer):
    
    init = RandomNormal(stddev = 0.02)   
    g = Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = init)(input_layer)
    g = BatchNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    g = Conv2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = init)(g)
    g = BatchNormalization(axis = -1)(g)
    g = Concatenate()([g, input_layer])
    
    return g


# --- GENERATOR ---

def define_generator(image_shape, rn_blocks = 6):
    init = RandomNormal(stddev = 0.02)
    in_image = Input(shape = image_shape)
    
    # #C7S1-64  -> kernel_size = 7, stride = 1, filters = 64
    g = Conv2D(filters = 64, kernel_size = (7, 7), strides = (1, 1), padding = 'same', kernel_initializer = init)(in_image)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #dk  -> 3x3 Conv-InstNorm-ReLU con k filtros y stride 2
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
        g = resnet_block(256, g)
    
    #uk -> 3x3 TransposedConv-InstanceNorm-ReLU, k filtros, stride = 0.5 *** TRANSPOSED CONV
    #u128 
    g = Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #u64
    g = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation('relu')(g)
    
    # #C7S1-3
    g = Conv2D(filters = 1, kernel_size = (7, 7), strides = (1, 1), padding = 'same', kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    output_image = Activation('tanh')(g) # tanh para valores [-1, 1], sigmoid para [0, 1]
    
    model = Model(inputs = in_image, outputs = output_image)
    return model

    
# ------------------ COMPOSITE MODEL -----------------

# entrena cada generador por separado - 1er generator: trainable, discriminator: non-trainable, 2do generator: non-trainable

def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True # 1er generator: trainable
    d_model.trainable = False # discriminator: non-trainable
    g_model_2.trainable = True # 2do generator : non-trainable
    
    # define 4 pérdidas de la arquitectura:
    
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
def generate_real_samples(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1)) # generate "real" class labels (1)
    return X, y

# generate a batch of (fake) images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1)) # create "fake" class labels (0)
    return X, y

# saves the generator model to files 
def save_models(step, g_model_AtoB, g_model_BtoA):
    filename1 = 'g_model_AtoB_%06d.h5' % (step + 1) # save the first generator model
    g_model_AtoB.save(filename1)
    
    filename2 = 'g_model_BtoA_%06d.h5' % (step + 1)
    g_model_BtoA.save(filename2)
    
    print('>Saved: %s and %s' % (filename1, filename2))
    
    
# periodically generate images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples = 5):
    
    X_in, _ = generate_real_samples(trainX, n_samples, 0) # select a sample of input images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
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
    filename1 = '%s_generated_plot_%06d.png' % (name, (step + 1))
    plt.savefig(filename1)
    plt.close()
    

# update image pool for fake images to reduce model oscillation
# update discriminators using a history of generated images
# rather than the ones produces by the latest generators.
# original paper recommends keeping an image buffer that stores
# the 50 previously created images
    
def update_image_pool(pool, images, max_size = 50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image) # stock the pool
            selected.append(image)
        elif random() < 0.5:
            selected.append(image) # use image, but don't add it to the pool
        else: #replace an existing image and use replaced image
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


# train cycleGAN models

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs = 1):
    # properties of training run
    n_epochs, n_batch = epochs, 1 # recommended in paper
    
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    
    # unpack dataset
    trainA, trainB = dataset
    
    # prepare image pool for fake images
    poolA, poolB = list(), list()
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples from each domain (A and B)
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        
        # generate a batch of fake samples using both B to A and A to B generators
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        
        # update fake images in the pool. (buffer of 50 images - paper)
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        
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
        print('Iteration>%d, dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' % (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        
        # evaluate the model performance periodically
        # if batch size (total of images) = 100, performance will be summarized after every 75th iteration
        
        if (i + 1) % (bat_per_epo * 1) == 0:
            # plot A -> B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B -> A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
            
        if (i + 1) % (bat_per_epo * 5) == 0: # 5
            # save the models
            # if batch size (total of images) = 100, model will be saved after every 75th iteration * 5 => 375 iterations
            
            save_models(i, g_model_AtoB, g_model_BtoA)
            
        
        
            
            
            
    
    