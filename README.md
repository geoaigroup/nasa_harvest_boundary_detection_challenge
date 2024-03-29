### :rocket: **4th Place Solution** : [Nasa Harvest Boundary Detection Challenge](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge)
<img
  src="/media/plot_visual.png"
  alt=""
  title="Model Output"
  style="display: inline-block; margin: 0 auto; max-width: 600px">
  
This is the 4th place solution for [NASA Harvest Field Boundary Detection Challenge](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge) on [Zindi](https://zindi.africa).
In this challenge, the goal was to classify crop field boundaries using time-series observations collected by PlanetScope. 
The hardest part of this challenge was the small amount of training data (57 images), and the fact that crop fields whose boundaries does not completely fall within the range of the image are left unlabelled.

### :rocket: Our Final Approach 
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

### :rocket: [Download the Data](https://mlhub.earth/data/nasa_rwanda_field_boundary_competition)
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
          
### :rocket: To Download our final pretrained models
The final models weights can be found here : 
 * [GCP Bucket](https://console.cloud.google.com/storage/browser/nasa_harvest_boundary_detection_challenge-solution)
 
You can download the model weights to a google colab notebook by using the following lines of code
```python
from google.colab import auth
auth.authenticate_user()
!gsutil -m cp -r gs://nasa_harvest_boundary_detection_challenge/nasa_rwanda_field_boundary_competition_final_weights/ /content/
```

### :rocket: To Run inference on the test set and create a final submission
```bash 
$ python3 ensemble_submission.py --input_dir '/final_pretrained_model_weights_directory' --data_dir './data' --save_file './geoai_final_submission.csv'
```
This will create a `**geoai_final_submission.csv**` file to be submitted to reproduce our results.

### :boom: Check more details about how to run inference with this colab notebook [`colab_demo_run.ipynb`](https://colab.research.google.com/github/geoaigroup/nasa_harvest_boundary_detection_challenge/blob/main/colab_demo_run.ipynb)

### :rocket: To Train our models from scratch
```bash 
$ python3 train_all.py --configs_dir './final_models_configs' --out_dir './final_models' --data_dir './data' --folds_path './folds.csv'
```
The trained models weights are saved inside a new directory `**final_models**`

### :rocket: Dependencies
* [Pytorch/TorchVision](https://github.com/pytorch/pytorch)
* [Segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [Albumentations](https://albumentations.ai/docs/getting_started/installation/)
* [Scikit-image](https://github.com/scikit-image/scikit-image)
* [Rasterio](https://github.com/rasterio/rasterio)
* [Scipy](https://github.com/scipy/scipy)
* [Pandas](https://github.com/pandas-dev/pandas)
* [Numpy](https://github.com/numpy/numpy)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [OpenCv](https://github.com/opencv/opencv-python)
* [Tqdm](https://github.com/tqdm/tqdm)
* [Einops](https://github.com/arogozhnikov/einops)
