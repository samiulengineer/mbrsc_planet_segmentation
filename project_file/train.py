from loss import *
from tensorflow.keras import backend as K
K.clear_session()

import os
import sys
import math
import glob
import time
import argparse
from model import get_model, get_model_transfer_lr, finetune_model
from metrics import get_metrics
from tensorflow import keras
from utils import SelectCallbacks, get_config_yaml, create_paths
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision


#tf.config.optimizer.set_jit(True)
#mixed_precision.set_global_policy('mixed_float16')

# Parsing variable
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--gpu")
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs")
parser.add_argument("--batch_size")
parser.add_argument("--index")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------

config = get_config_yaml('config.yaml', vars(args))
create_paths(config)



# Setup train strategy Muli-GPU or single-GPU
# ----------------------------------------------------------------------------------------------

#strategy = set_gpu(config['gpu'])


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------

print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['x_train_dir'])))



# Dataset
# ----------------------------------------------------------------------------------------------

train_dataset, val_dataset = get_train_val_dataloader(config)


# enable training strategy
#with strategy.scope():
metrics = list(get_metrics(config).values())
adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])
#adam = tf.keras.mixed_precision.LossScaleOptimizer(adam)


# create dictionary with all custom function to pass in custom_objects
custom_obj = get_metrics(config) 
custom_obj['loss'] = focal_loss()
    
if (os.path.exists(config["checkpoint_path"])) and config['transfer_lr']:
    print("Build model for transfer learning..")
    # load model and compile
    model = load_model(config["checkpoint_path"], custom_objects=custom_obj, compile = True)

    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)

elif (os.path.exists(config["checkpoint_path"])) and config['finetune']:
    print("Build model for Finetune..")
    # load model and compile
    model = load_model(config["checkpoint_path"], custom_objects=custom_obj, compile = True)

    model = finetune_model(model, config['num_classes'])
    model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)

else:
    if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))):
        print("Resume training from model checkpoint {}...".format(config['load_model_name']))
        # load model and compile
        model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile = True)

    else:
        model = get_model(config)
        model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)


# Set up Callbacks
# ----------------------------------------------------------------------------------------------

loggers = SelectCallbacks(val_dataset, model, config)
model.summary()


# fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose = 1, 
                    epochs = config['epochs'],
                    validation_data = val_dataset, 
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    )
print(time.time() - t0)
#model.save('/content/drive/MyDrive/CSML_dataset/model/my_model.h5')