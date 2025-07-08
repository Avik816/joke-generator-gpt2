'''
This script describes the model's training parameters.
It sets up the model's callbacks to prevent overfitting and make the model generalize better.
The callbacks also have been provided with a logger file which will log in all the model's hyper-params.
'''

import datetime
import tensorflow as tf
from transformers import GPT2Tokenizer
from .model import setup_model
from ..data_for_model.model_data_2 import create_model_dataset
from .. import CONFIG
from ..plotting.model_learning_curve import plot_learning_curve
# import math


# Loading the model to train
model = setup_model(CONFIG.MODEL_NAME, CONFIG.LEARNING_RATE)

# Loading tokenizer and defining pad token
tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad_token, so we use eos_token of GPT2

train_dataset, val_dataset = create_model_dataset(tokenizer)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Stops the model if the validation loss is not improving for 2 epochs.
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Creates a checkpoint to save the model.
# Model is saved in its entireity (architecture, weights etc.).
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{CONFIG.OUTPUT_DIR_MODEL_WEIGHTS}/model_weights_{timestamp}_epoch-{{epoch:02d}}_val_loss-{{val_loss:.6f}}.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# When the model's validation loss is not improving for the said epochs then,
# this callback will reduce the Learning Rate by the Factor and then wait for another (in this case 2) epochs.
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=2,
    mode='min',
    min_lr=1e-6,
    verbose=1
)

# Logs everything.
csv_logger_cb = tf.keras.callbacks.CSVLogger(
    filename=f'logs/training_log_{timestamp}.csv',
    append=True
)

def training_gpt2_small():
    '''with open(CONFIG.TRAIN_PATH, 'r', encoding='utf-8') as fp:
        num_samples = sum(1 for line in fp if "<|endoftext|>" in line)
    TRAIN_STEPS = math.ceil(num_samples / CONFIG.BATCH_SIZE)

    num_samples = 0
    with open(CONFIG.VAL_PATH, 'r', encoding='utf-8') as fp:
        num_samples = sum(1 for line in fp if "<|endoftext|>" in line)
    VAL_STEPS = math.ceil(num_samples / CONFIG.BATCH_SIZE)'''

    # Training the model.
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG.EPOCHS,
        # steps_per_epoch=TRAIN_STEPS,
        # validation_steps=VAL_STEPS,
        callbacks=[earlystop_cb, checkpoint_cb, reduce_lr_cb, csv_logger_cb]
    )

    # Saving model
    model.save_pretrained(f'{CONFIG.OUTPUT_DIR_MODEL}/joke_generator_v1')
    tokenizer.save_pretrained(CONFIG.OUTPUT_DIR_TOKENIZER)

    # Plotting the learning curve of the present model
    plot_learning_curve(history=history, timestamp=timestamp)