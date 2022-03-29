import os
import sys


from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf, setup_model
from utils import has_splits, get_log_dir, extract_toml
from data_utils import generate_train_val, gen_splits, inspect_settings


# Parse arguments from script call
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="Name of experiment to run", type=str)
parser.add_argument("--load_model", help="load model", default=False, action='store_true')
parser.add_argument("--epochs", help="# of training epochs", default=50, type=int)
parser.add_argument("--inspect", help="inspect model settings", default=0, type=int)

args = parser.parse_args()
EXPERIMENT = args.experiment
LOAD_MODEL = args.load_model
EPOCHS = args.epochs
INSPECT = args.inspect


# Initialize training parameters
batch_size = 128


if INSPECT:
	inspect_settings(EXPERIMENT)
	exit(0)


# Get training and validation data
if not has_splits(EXPERIMENT):
	gen_splits(EXPERIMENT)

train_dataset, val_dataset, maps = generate_train_val(EXPERIMENT, batch_size)


# Initialize model saving
log_dir = get_log_dir(EXPERIMENT)
writer = tf.summary.create_file_writer(log_dir)


writer = keras.callbacks.TensorBoard(log_dir, update_freq=1)
early = tf.keras.callbacks.EarlyStopping(monitor="val_policy_acc",
        patience=10,
        verbose=0,
        restore_best_weights=True
)
callbacks = [writer, early]


# Initialize, build, and compile model


toml_data = extract_toml(EXPERIMENT)

model = setup_model(EXPERIMENT,batch_size,maps,load=LOAD_MODEL);
model.summary()


# Train model
model.fit(
	train_dataset,
	validation_data = val_dataset,
	validation_freq = 1,
	callbacks = callbacks,
	epochs = EPOCHS,
	verbose = 1
)

model.evaluate(val_dataset)


# Save Model
model.save_weights(log_dir + 'model/weights')
