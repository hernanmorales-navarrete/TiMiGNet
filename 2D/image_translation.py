#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from models import *

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def image_preprocessing(image, min_val, max_val):
    
    img = tiff.imread(image)
    img_shape = img.shape
    
    img = np.reshape(img, newshape = -1)
    img = cv2.normalize(img, None, alpha = min_val, beta = max_val, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = np.reshape(img, newshape = img_shape)
    
    return img




def image_postprocessing(prediction):
    to_uint16 = (prediction + 1) / 2.0
    to_uint16 = (to_uint16 * 65535).astype(np.uint16)
    
    return to_uint16
    
    
    

# =============================================================================
# TRANSLATE 2D IMAGE WITH PATCHIFY
# =============================================================================


def translate_patchify_2D(image, patch_shape, patch_step): #patch_shape -> tuple. patch_step-> single value

    dim_0 = int((np.ceil(image.shape[0] / patch_shape[0]) * patch_shape[0]) - image.shape[0])
    dim_1 = int((np.ceil(image.shape[1] / patch_shape[0]) * patch_shape[0]) - image.shape[1])
    dim_2 = int((np.ceil(image.shape[1] / patch_shape[0]) * patch_shape[0]) - image.shape[2])

    npad = ((0,0), (0, dim_1), (0, dim_2)) # 16
    img3_3d = np.pad(image, pad_width = npad, mode = 'constant', constant_values = 0)
    
    prediction = []
    img_number = 1
    
    for img in img3_3d:
        patch_number = 1
        predicted_patches = []

        patches = patchify(img, patch_shape, step = patch_step)
        
        for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single_patch = patches[i, j, :, :]
                    single_patch = np.array(single_patch)
                    single_patch = np.expand_dims(single_patch, 2)
                    single_patch_shape = single_patch.shape[:2]
                    if single_patch_shape == patch_shape:
                        single_patch_input = np.expand_dims(single_patch, 0)
                        #single_patch_prediction = model.predict(single_patch_input)
                        single_patch_prediction = single_patch_input # BORRAR ESTA LINEA Y AGREGAR MODELO EN LA ANTERIOR. TAMBIEN AGERGAR PARAMETRO A FUNCION.
                        predicted_patches.append(single_patch_prediction)
        
                        print('image: {} — patch: {}'.format(img_number, patch_number))
                        patch_number +=1
                    else: continue
                
        predicted_patches = np.array(predicted_patches)
        predicted_patches = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_shape[0], patch_shape[1]))
        prediction.append(unpatchify(predicted_patches, img.shape))
        img_number += 1
        
    prediction_ = np.array(prediction)
    prediction_ = prediction_[0: image.shape[0], 0: image.shape[1], 0: image.shape[2]]
        
    return prediction_



def translate_patchify_3D(image, patch_shape, patch_step):
    
    # PADDING
    
    dim_0 = int((np.ceil(image.shape[0] / patch_shape[0]) * patch_shape[0]) - image.shape[0])
    dim_1 = int((np.ceil(image.shape[1] / patch_shape[0]) * patch_shape[0]) - image.shape[1])
    dim_2 = int((np.ceil(image.shape[1] / patch_shape[0]) * patch_shape[0]) - image.shape[2])
    
    npad = ((0, dim_0), (0, dim_1), (0, dim_2))
    img = np.pad(image, pad_width = npad, mode = 'constant', constant_values = 0)
    
    # HASTA AQUI VA BIEN # return img
    
    # EXTRACT 3D PATCHES
    
    patches = patchify(img, patch_size = patch_shape, step = patch_step)
    patches_ = np.reshape(patches, (-1, patch_shape[0], patch_shape[1], patch_shape[2]))


    # APPLY MODEL PREDICTION
    
    prediction = []
    counter = 1
    for patch in patches_:
        patch = np.expand_dims(patch, axis = 0)
        prediction.append(patch)
        #prediction.append(model.predict(patch)) # BORRAR LINEA ANTERIOR Y AGREGAR MODELO A PARAMETRO DE FUNCION
        print('patch %d of %d' % (counter, patches_.shape[0]))
        counter += 1
        
    prediction_ = np.array(prediction)
    prediction_ = np.reshape(prediction_, (patches.shape[0], patches.shape[1], patches.shape[2], patch_shape[0], patch_shape[1], patch_shape[2]))
    prediction_ = unpatchify(prediction_, img.shape)
    prediction_ = prediction_[0: image.shape[0], 0: image.shape[1], 0: image.shape[2]]
    
    return prediction_





# =============================================================================
# TRANSLATE 2D IMAGE USING SMOOTH TILED PREDICTIONS
# =============================================================================

def translate_smooth(image, model, window_size):
    
    image_shape = image.shape
    npad = ((0, 0), (window_size, window_size), (window_size, window_size))
    image = np.pad(image, pad_width = npad, mode = 'constant', constant_values = 0)
    
    translated = []
    n_img = 1
    for img in image:
        print('image: ', n_img)
        img = np.expand_dims(img, 2)
        
        predictions_smooth = predict_img_with_smooth_windowing(img,
                                                               window_size = window_size,
                                                               subdivisions = 2,
                                                               nb_classes = 1,
                                                               pred_func=(lambda beta: model.predict((beta))))
        translated.append(predictions_smooth)
        n_img += 1
    translated_np = np.squeeze(np.array(translated))
    translated_np = translated_np[:, window_size:image_shape[1] + window_size, window_size:image_shape[2] + window_size]
    return translated_np



#%%
# IMAGE AND MODEL PATHS

img = '/home/nicolas/Documentos/GANs/GANs_MATLAB/memb_img3.tif'
cust = {'InstanceNormalization': InstanceNormalization}
gen_BtoA_norm = load_model('./g_model_BtoA_2D_500000.h5', cust)


image = image_preprocessing(img, -1, 1)

translated = translate_smooth(image, gen_BtoA_norm, 128)

# STEP 1: call "image_preprocessing" function. Returns the preprocessed image in the same way it was preprocessed for model training.

# STEP 2: call "translate_patchify_2D" function. Returns the translation using patchify.

# STEP 3: call "image_postprocessing" function. It will return the image as uint16.

# STEP 4: call "save_image" function. It will save the image as a .tif file in the specified path.
del image

translated_uint16 = image_postprocessing(translated)

del translated

tiff.imwrite('./memb_img3_translated_v08_2D_version3.tif', translated_uint16)
#%%
'''
img_shape = 200
patch_shape = 32

f = int((np.ceil(img_shape/patch_shape)*patch_shape) - img_shape)

print(f)

'''






