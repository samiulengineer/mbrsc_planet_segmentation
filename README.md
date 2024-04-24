# Image Segmentation

## Introduction

In this project we are tring to predict corresponding mask or segment image from given satelite image or feature. Image segmentation model can be use to extract real life objects from images, blur background, self driving car and other image processing tasks.


## Logger Path

During model training following paths will be created automatically.

1. csv_logger: model metrics will be saved in csv format
2. logs: tensorboard logger file will be saved here
3. model: model checkpoint will be saved here
4. prediction: validation and testing prediction will be saved here

## Dataset

For our practice we use [Semantic segmentation of aerial imagery](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery) data from kaggle. The dataset contains 72 images which group into 8 larger tiles for 6 classes each tiles contain 9 images of same dimension or close to each other. The classes are as follows:

| Name       | R   | G   | B   | Color                                                                                              |
| ---------- | --- | --- | --- | -------------------------------------------------------------------------------------------------- |
| Building   | 60  | 16  | 152 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_building.png" /></p>   |
| Land       | 132 | 41  | 246 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_land.png" /></p>       |
| Road       | 110 | 193 | 228 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_road.png" /></p>       |
| Vegetation | 254 | 221 | 58  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_vegetation.png" /></p> |
| Water      | 226 | 169 | 41  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_water.png" /></p>      |
| Unlabeled  | 155 | 155 | 155 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_unlabeled.png" /></p>  |

As each tile contains images of different dimension we pachify the images into fixed dimension 256 X 256 and save those in directory as mention in setup section below. The pachify image dimension can be changed by `height` and `weight` variable inside the `config.yaml` file.

Image             |  Mask
:-------------------------:|:-------------------------:
![Alternate text](/readme/Tile_4_patch_2_0_image_part_002.jpg)  |  ![Alternate text](/readme/Tile_4_patch_2_0_image_part_002.png)
 

## Downloasd Dataset

Download dataset from [here](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fhumansintheloop%2Fsemantic-segmentation-of-aerial-imagery&sa=D)

## Models

The following models are available in this repository. We train all the models for our project.

* [UNET](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
* [UNET++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)
* [DNCNN](https://ieeexplore.ieee.org/document/7839189)
* [VNET](https://arxiv.org/abs/1606.04797)
* [U2NET](https://paperswithcode.com/paper/u-2-net-going-deeper-with-nested-u-structure)

## Setup

Before start training check the variable inside config.yaml i.e. `height`, `in_channels`. Asuming that you have images in `image.jpg` and masks in `mask.png` format and data directory looks like following you can directly go for [training](##Training) skip the below part.

```
--dataset
    --Tile1
        --images
            --image.png
            ..
        --masks
            --mask.png
            ..
    --Tile2
        --images
            --image.png
            ..
        --masks
            --mask.png
            ..
    ..
```

For train model in a new dataset you need to follow the following directory format.

```
--dataset
    --train
        --img
            --image.png
            ..
        --mask
            --mask.png
            ..
    --valid
        --img
            --image.png
            ..
        --mask
            --mask.png
            ..
    --test
        --img
            --image.png
            ..
        --mask
            --mask.png
            ..
```

## Training

Validation handle automatically during taining and after each (this can be control by `val_pred_plot` variable inside `config.yaml`) epoch model will save single figure that contains image, mask, pred_mask and accuracy of pred_mask. While training we use CSVLogger(save metrics and loss after each epoch in .csv file), ModelCheckpoint(update/save best model weight after each epoch), TensorBoard(display graph), LearningRateScheduler(decrease learning rate based on model state), EarlyStopping(if accuracy does not increase for certain time it will stop training) and custom SelectCallbacks(save validation plot based on `val_pred_plot` variable) keras callbacks.

### Training on Single GPU

```
cd project
python train.py --gpu "0" \
    --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_DATASET_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
```

### Training on Multi-GPU

```
cd project
python train.py --gpu "0,1,2" \
    --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_DATASET_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
```

### Resume Training

To resume training from a checkpoint that's train using this repository you need to change `model_name` and `load_model_name` variable inside `config.yaml` file. Rest will be same as training section.

## Transfer Learning

Use checkpoint for transfer learning that's train using this repository you need to change `model_name`, `load_model_name` and `transfer_lr` variable inside `config.yaml` file. Rest will be same as training section.

**Note: Remember for transfer learning you are only able to train the last layer.**

## Testing

For testing model on you test dataset use the following code

```
cd project
python train.py --gpu "0" \
    --dataset_dir YOUR_DATASET_DIR \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --load_model_dir MODEL_DIR \
```

## Result

![Alternate text](/readme/prediction_result.png)