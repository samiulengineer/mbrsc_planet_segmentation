import os 
import cv2
import math
import glob
import shutil
import random
import pathlib
from loss import *
import numpy as np
from PIL import Image
import tensorflow as tf
import albumentations as A
from patchify import patchify
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence

# unpack labels        
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#') 
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#') 
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#') 
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155




def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """


    label_seg = np.zeros(label.shape,dtype = np.uint8)
    label_seg [np.all(label == Building, axis = -1)] = 0
    label_seg [np.all(label == Land, axis = -1)] = 1
    label_seg [np.all(label == Road, axis = -1)] = 2
    label_seg [np.all(label == Vegetation, axis = -1)] = 3
    label_seg [np.all(label == Water, axis = -1)] = 4
    label_seg [np.all(label == Unlabeled, axis = -1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg




def read_img(directory, rgb=False, norm=False):
    """
    Summary:
        read image with opencv and normalize the feature
    Arguments:
        directory (str): image path to read
        rgb (bool): convert BGR to RGB image as opencv read in BGR format
    Return:
        numpy.array
    """


    if rgb:
        return cv2.cvtColor(cv2.imread(directory, 1), cv2.COLOR_BGR2RGB) # read and convert from BGR to RGB
    elif norm:
        return np.array(cv2.imread(directory, 1)/255) # MinMaxScaler can be used for normalize
    else:
        return np.array(cv2.imread(directory, 1)) 




def move_images_mask_from_tile(config):
    """
    Summary:
        Pacify images and masks after read from Tile folders.
        Save each pacify images and masks into images and masks folder under dataset_dir
    Arguments:
        config (dict): Configuration directory
    Return:
        return directory of saved patchify images and masks.
    """


    # create MinMaxScaler object
    scaler = MinMaxScaler((0.0,0.9999999)) # sklearn sometime give value greater than 1
    os.mkdir((config['dataset_dir']+'imgs')) # creating image directory under dataset directory
    img_patch_dir = config['dataset_dir']+'imgs/'

    for path, subdirs, files in os.walk(config['dataset_dir'], topdown=True):
        # print(sorted(subdirs))
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':   #Find all 'images' directories
            images = os.listdir(path)#List of all image names in this subdirectory
            images = sorted(images)
            for i, image_name in enumerate(images):  
                
                if image_name.endswith(".jpg"):   #Only read jpg images...
                
                    image = read_img((path+"/"+image_name), norm=False)  #Read each image as BGR
                    SIZE_X = (image.shape[1]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    image = np.array(image)             
        
                    #Extract patches from each image
                    print("Now patchifying image:", path+"/"+image_name)
                    patches_img = patchify(image, (config['patch_size'], config['patch_size'], 3), step=config['patch_size'])  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:]
                            
                            #Use minmaxscaler instead of just dividing by 255.
                            scaler.fit(single_patch_img.reshape(-1, single_patch_img.shape[-1]))
                            single_patch_img = scaler.transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                            save_path = path.replace("/images","").split("/")[-1]
                            save_path = img_patch_dir+save_path.replace(" ","_")+"_patch_{}_{}".format(i, j)+"_"+image_name
                            plt.imsave(save_path, single_patch_img)  

    os.mkdir((config['dataset_dir']+'msks')) # creating masks directory under dataset directory
    mask_patch_dir = config['dataset_dir']+'msks/'
    for path, subdirs, files in os.walk(config['dataset_dir'], topdown=True):
            
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':   #Find all 'images' directories
            masks = os.listdir(path)  #List of all image names in this subdirectory
            masks = sorted(masks)
            for i, mask_name in enumerate(masks): 

                if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                
                    mask = read_img((path+"/"+mask_name), rgb=True)  #Read each image as Grey (or color but remember to map each color to an integer)
                    SIZE_X = (mask.shape[1]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)             
        
                    #Extract patches from each image
                    print("Now patchifying mask:", path+"/"+mask_name)
                    patches_mask = patchify(mask, (config['patch_size'], config['patch_size'], 3), step = config['patch_size'])  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
                            save_path = path.replace("/masks","").split("/")[-1]
                            save_path = mask_patch_dir+save_path.replace(" ","_")+"_patch_{}_{}".format(i, j)+"_"+mask_name
                            plt.imsave(save_path, single_patch_mask)
    return img_patch_dir, mask_patch_dir




def data_split(images, masks, config):
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
        config (dict): Configuration directory
    Return:
        return the split data.
    """


    x_train, x_rem, y_train, y_rem = train_test_split(images, masks, train_size = config['train_size'])
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)
    return x_train, y_train, x_valid, y_valid, x_test, y_test




def move_image(dir_name, directories):
    """
    Summary:
        create new directory and move files into
        new created directory
    Arguments:
        dir_name (str): directory to create
        directories (list): directory where file will move
    Output:
        directory of move image
    """


    os.mkdir(dir_name)
    for i in range(len(directories)):
        shutil.move(directories[i], dir_name) # move directory[i] file to dir_name



def data_split_and_foldering(config):
    """
    Summary:
        split data and foldering using above helper functions
    Arguments:
        config (dict): Configuration directory
    Output:
        data split and save in folders
    """


    print("Start data split and foldering....")
    img_dir, mask_dir = move_images_mask_from_tile(config) # save pachify images and masks directory

    # sorting so that there is no mis-match between image and mask
    image_dataset = sorted(glob.glob((img_dir+'*.jpg')))
    mask_dataset = sorted(glob.glob((mask_dir+'*.png')))


    print("Total number of images : {}".format(len(image_dataset)))
    print("Total number of masks : {}".format(len(mask_dataset)))

    # split data
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(image_dataset, mask_dataset, config)

    # directory create and move img
    os.mkdir((config['dataset_dir']+'train'))
    move_image((config['dataset_dir']+'train/img'), x_train)
    move_image((config['dataset_dir']+'train/mask'), y_train)
    
    os.mkdir((config['dataset_dir']+'valid'))
    move_image((config['dataset_dir']+'valid/img'), x_valid)
    move_image((config['dataset_dir']+'valid/mask'), y_valid)
    
    os.mkdir((config['dataset_dir']+'test'))
    move_image((config['dataset_dir']+'test/img'), x_test)
    move_image((config['dataset_dir']+'test/mask'), y_test)
    
    print("Complete data split and foldering.")



def transform_data(label, num_classes):
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """


    return to_categorical(rgb_to_2D_label(label), num_classes = num_classes)



# Data Augment class
# ----------------------------------------------------------------------------------------------

class Augment(tf.keras.layers.Layer):
    def __init__(self, batch_size, ratio=0.3, seed=42):
        super().__init__()
        """
        Summary:
            initialize class variables
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
        Return:
            class object
        """


        self.ratio=ratio
        self.aug_img_batch = math.floor(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),
                    A.GaussNoise(p=0.5)])

    def call(self, feature_dir, label_dir):
        """
        Summary:
            randomly select a directory and augment data 
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """
        
        
        
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []
        
        # spatial transformation
        for i in aug_idx:
            img = read_img(feature_dir[i], norm=True)
            mask = read_img(label_dir[i], rgb=True)
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented['image'])
            labels.append(augmented['mask'])
        return features, labels



# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, batch_size, transform_fn=None, num_class=None, augment=None, weights=None):
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            batch_size (int): how many data to pass in a single step
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
        Return:
            class object
        """


        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights



    def __len__(self):
        """
        return total number of batch to travel full dataset
        """


        return math.ceil(len(self.img_dir) / self.batch_size)


    def __getitem__(self, idx):
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """


        # get index for single batch
        batch_x = self.img_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        imgs = []
        tgts = []
        for i in range(len(batch_x)):
          imgs.append(read_img(batch_x[i], norm=True))
          # transform mask for model
          if self.transform_fn:
            tgts.append(self.transform_fn(read_img(batch_y[i], rgb=True), self.num_class))
          else:
            tgts.append(read_img(batch_y[i], rgb=True))
        
        # augment data using Augment class above if augment is true
        if self.augment:
            aug_imgs, aug_masks = self.augment(self.img_dir, self.tgt_dir) # augment images and mask randomly
            imgs = imgs+aug_imgs

            # transform mask for model
            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts+aug_masks
        
        tgts = np.array(tgts)
        imgs = np.array(imgs)        

        if self.weights != None:
            y_weights = tf.gather(self.weights, indices=tf.cast(tgts, tf.int32))#([self.paths[i] for i in indexes])

            return tf.convert_to_tensor(imgs), y_weights

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)
    
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    

    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """



        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        
        imgs = []
        tgts = []
        imgs.append(read_img(self.img_dir[idx], norm=True))
        
        # transform mask for model
        if self.transform_fn:
            tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], rgb=True), self.num_class))
        else:
            tgts.append(read_img(self.tgt_dir[idx], rgb=True))
        return np.array(imgs), np.array(tgts), idx


def get_train_val_dataloader(config):
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """


    if not (os.path.exists(config['x_train_dir'])):
      data_split_and_foldering(config)
    else:
        print("Loading image and mask directories.....")

    x_train = sorted(glob.glob((config['x_train_dir']+'*.jpg')))
    x_valid = sorted(glob.glob((config['x_valid_dir']+'*.jpg')))
    y_train = sorted(glob.glob((config['y_train_dir']+'/*.png')))
    y_valid = sorted(glob.glob((config['y_valid_dir']+'/*.png')))

    print("x_train Example : {}".format(len(x_train)))
    print("x_valid Example : {}".format(len(x_valid)))
        
    print("y_train Example : {}".format(len(y_train)))
    print("y_valid Example : {}".format(len(y_valid)))

    # create Augment object if augment is true
    if config['augment']:
        augment_obj = Augment(config['batch_size'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None
    
    if config['weights']:
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None

    # create dataloader object
    train_dataset = MyDataset(x_train, y_train, batch_size=n_batch_size, 
                              transform_fn=transform_data, num_class=config['num_classes'], 
                              augment=augment_obj, weights=weights)
    val_dataset = MyDataset(x_valid, y_valid, batch_size=config['batch_size'], transform_fn=transform_data, num_class=config['num_classes'])
    return train_dataset, val_dataset

def get_test_dataloader(config):
    """
    Summary:
        read test image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        test dataloader
    """


    if not (os.path.exists(config['x_test_dir'])):
        print("Directory missing!!!!")
    else:
        print("Loading test data directories....")
    
    x_test = sorted(glob.glob((config['x_test_dir']+'*.jpg')))
    y_test = sorted(glob.glob((config['y_test_dir']+'/*.png')))
    
    print("x_test Example : {}".format(len(x_test)))
    print("y_test Example : {}".format(len(y_test)))

    # create dataloader object
    test_dataset = MyDataset(x_test, y_test, batch_size=config['batch_size'], transform_fn=transform_data, num_class=config['num_classes'])
    
    return test_dataset

def find_class_distribution(config):
    y_train = sorted(glob.glob((config['y_train_dir']+'/*.png')))

    total_pixel = 0
    build = 0 # 0
    land = 0 # 1
    road = 0 # 2
    vege = 0 # 3
    water = 0 # 4
    unla = 0 # 5
    # https://stats.stackexchange.com/questions/355248/how-to-set-class-weights-for-multi-class-image-segmentation

    for p in y_train:
        transf = transform_data(read_img(p, rgb=True), 6)
        build += sum(transf[:,:,0].reshape(-1))
        land += sum(transf[:,:,1].reshape(-1))
        road += sum(transf[:,:,2].reshape(-1))
        vege += sum(transf[:,:,3].reshape(-1))
        water += sum(transf[:,:,4].reshape(-1))
        unla += sum(transf[:,:,5].reshape(-1))
        total_pixel += (transf.shape[0]*transf.shape[1])
    print("Total Training: ", len(y_train))
    print("Building percentage: ", (build/total_pixel)*100)
    print("Land percentage: ", (land/total_pixel)*100)
    print("Road percentage: ", (road/total_pixel)*100)
    print("Vegetation percentage: ", (vege/total_pixel)*100)
    print("Water percentage: ", (water/total_pixel)*100)
    print("Unlabeled percentage: ", (unla/total_pixel)*100)