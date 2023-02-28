### [Nasa Harvest Boundary Detection Challenge](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge)
This is the 4th place solution for NASA Harvest Field Boundary Detection Challenge on Zindi.
In this challenge, the goal was to classify crop field boundaries using multispectral observations collected by PlanetScope. 
The hardest part of this challenge was the small amount of training data (57 images), and the fact that crop fields whose boundaries does not completely fall within the range of the image are left unlabelled.

### Our Final Approach 
In a nutshell, we tried several ideas, but here is a list of the things that worked best:
 1. Unet-Like encoder/decoder architecture with 3D-convolutions at the out encoder blocks. Given a time-series of images, each image is processed alone in the encoder. The features from every block for all images in a time-series are then stacked together and a 3D convolution is applied to transform from 3D to 2D. These transformed features are then processed normally by the decoder.
 2. Encoders used : EfficientNetV2_S - EfficientNetV2_M - EfficientNetV2_B2 - SKResnet34
 3. Augmentations : Flips - Rotations - Mixup (applied twice).
 4. Random Masking of the output to ignore loss computation on a portion of the image. This helps prevent overfitting.
 5. Loss Function : Dice + BCE
 6. Knowledge Distillation : at last we trained 4 models using the teacher/student setting.
 7. Ensembling via simple averaging.
 8. Noise Removal.
 9. Test Time Augmentations(V/H Flips and 180 degree rotate)

### [Download the Data](https://mlhub.earth/data/nasa_rwanda_field_boundary_competition)
 1. Download the train images, train labels and test images of this challenge and extract them in the /data folder
 2. The data folder now should look like this :
``` 
/data 
|______ nasa_rwanda_field_boundary_competition_source_train
|______ nasa_rwanda_field_boundary_competition_labels_train
|______ nasa_rwanda_field_boundary_competition_source_test
````
 3. Rename the folders to 'train_imgs', 'train_labels' and 'test_imgs'
 
 4. The data folder now should look like this : 
```
 /data
|______ train_imgs
|______ train_labels
|______ test_imgs
```
          

### To Train our models from scratch
```bash 
$ python3 train_all.py --configs_dir './final_models_configs' --out_dir './final_models' --data_dir './data' --folds_path './folds.csv'
```
The trained models weights are saved inside a new directory './final_models'

### To Run inference on the test set and create a final submission
```bash 
$ python3 ensemble_submission.py --input_dir './final_models' --data_dir './data'
```
This will create a final_submission.csv file to be submitted to reproduce our results.

