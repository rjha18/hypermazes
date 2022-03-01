import os
import sys


from tensorflow import keras
import tensorflow as tf
import argparse

from model import rlf
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
e_sz = [64,64,16]
f_sz = [32,32,1]
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
callbacks = [keras.callbacks.TensorBoard(log_dir, update_freq=1)]


# Initialize, build, and compile model


toml_data = extract_toml(EXPERIMENT)

model = rlf(e_sz,f_sz,batch_size,maps,method=toml_data['hypernet'],lr=1e-4,writer=writer)
model.build([(batch_size),(batch_size, 4),(batch_size, 2)])

print(model.summary())

if LOAD_MODEL:
	model.load_weights(log_dir + 'model/weights')\
		 .expect_partial()


loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(
	optimizer=keras.optimizers.Adam(1e-5),
	loss=loss_fn,
	metrics=[loss_fn]
)


# Train model
model.fit(
	train_dataset,
	validation_data = val_dataset,
	callbacks = callbacks,
	epochs = EPOCHS,
	verbose = 1
)

model.evaluate(val_dataset)


# Save Model
model.save_weights(log_dir + 'model/weights')
