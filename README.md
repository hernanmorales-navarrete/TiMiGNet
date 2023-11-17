# TiMiGNe
## Requirements
### Libraries:
* Python 3.9.7
* NumPy 1.19.5
* Pandas 1.3.4
* OpenCV 4.6.0
* Matplotlib 3.6.2
* Tensorflow 2.7.0
* Keras 2.7.0
* Tifffile 2022.10.10
* Patchify 0.2.3
* EMPatches 0.2.2


### Scripts:
* instancenormalization
* smooth_tiled_predictions

## Credits:
* TiMiGNet code based on the implementations by Sreenivas Bhattiprolu and Jason Brownlee. https://github.com/bnsreenu/python_for_microscopists/blob/master/253_254_cycleGAN_monet2photo/254-cycleGAN_model.py
* Instance Normalization layer: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
* Smooth Blend Image Patches: https://github.com/Vooban/Smoothly-Blend-Image-Patches
* Extract and Merge Image Patches (EMPatches): https://github.com/Mr-TalhaIlyas/EMPatches

# How to use

## Training


Set the path to both images into the variables a_0 and b_0 for domain A and domain B, respectively. Define a new variable and call the function “generate_dataset”. This function will generate a dataset using patches extracted from both images. This function takes the following as input:
* **model_type:** ‘2d’ for 2D-model or ‘3d’ for 3D-model.
* **images:** a list containing both images (paths).
* **patch_shape:** the size of a single patch 
* **patch_step:** the step size between patches. Aka ‘overlap’. Same as patch_shape for non-overlapping patches.
* **n_samples:** number of patches per image.
* **seed_a:** seed for image a
* **seed_b:** seed for image b
* **shuffle_:** boolean. Will shuffle the order of patches according to seed_a and seed_b. Set to True by default. 

Example
```python

a = "/path_to_image_domain_A.tif"
b = "/path_to_image_domain_B.tif"

dataset = generate_dataset(model_type = ‘2d’,
			images = [a, b],
			patch_shape = (64, 64),
			patch_step = 64,
			n_samples = 500, 
			seed_a = 27,
			seed_b = 57,
			shuffle_ = True)
```

The majority of models used in this project were trained using two images per domain. Just generate two different datasets, and then merge them with the function “merge_datasets”:

```python
a_0: "/path_to_image_one_domain_a.tif"
a_1: "/path_to_image_two_domain_a.tif"

b_0: "/path_to_image_one_domain_b.tif"
b_1: "/path_to_image_two_domain_b.tif"

images_0 = [a_0, b_0]
images_1 = [a_1, b_1]

dataset_0 = generate_dataset(images = images_0)
dataset_1 = generate_dataset(images = images_1)

dataset = merge_datasets(datasets = [dataset_0, dataset_1],
				seed_a = 27, seed for dataset 0
				seed_b = 57, seed for dataset 1
				shuffle_ = True) # will shuffle datasets again. Useful if images are paired.
```


Train the model.

Call the function “train_model_2D” or “traing_model_3D” on a new variable. This function will take the following as input:
* **dataset:** two dimensional array previously generated.
* **batch_size:** set to 1 by default.
* **epochs:** number of epochs.
* **save_freq:** int.

```python
training = train_model_2D(dataset = dataset,
                          batch_size = 1,
                          epochs = 100,
                          save_freq = 5)
                        
```
Two folders will be generated in the working directory:
* **'example_images’:** will save sample prediction images during training every *“save_freq”* epochs. 
* **‘models’:** will save two models every *“save_freq”* epochs, one for GeneratorAtoB and one for GeneratorBtoA. 
Once the training process is done, the execution time will be printed on the console and a *“training_losses.csv”* file, which keeps track of the different losses obtained during training, will be created in the working directory. These losses are saved four times per epoch. 


## Evaluation

### 2D models. 

Use the function “predict_2D”.

```python
prediction = predict_2D(input_image_path, # path to image
			   output_name, # output filename. Will be saved in the working directory.
			   model_path, # path to model (.h5 file)
			   patch_size, # size of patch (single value, square))
```

### 3D models. 

Use the function “predict_3D”.

```python
prediction = predict_3D(input_image_path,  # path to image
			    output_name, # output filename. Will be saved in the working directory.
	 		    model_path, # path to model (.h5 file)
			    patch_size, # size of patch (single value)
			    overlap_pct, # overlap percentage. Default set to 0.125 for 8 pix overlap for patch size 64.)
          
 ```
A .tif image called *"output_name.tif"* will be generated in the working directory.
