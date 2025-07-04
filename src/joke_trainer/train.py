import datetime
import tensorflow as tf
from .model import setup_model
from ..data_for_model.model_data_2 import create_model_dataset
from .. import CONFIG
from ..plotting.model_learning_curve import plot_learning_curve
import math


model = setup_model(CONFIG.MODEL_NAME, CONFIG.LEARNING_RATE)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_dataset, val_dataset = create_model_dataset()

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{CONFIG.OUTPUT_DIR_MODEL}/best_model_{timestamp}_epoch-{{epoch:02d}}_loss-{{val_loss:.4f}}.keras',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=2,
    mode='min',
    min_lr=1e-6,
    verbose=1
)

csv_logger_cb = tf.keras.callbacks.CSVLogger(
    filename=f'logs/training_log_{timestamp}.csv',
    append=True
)

def training_gpt2_small():
    with open(CONFIG.TRAIN_PATH, 'r', encoding='utf-8') as fp:
        num_samples = sum(1 for line in fp if "<|endoftext|>" in line)
    TRAIN_STEPS = math.ceil(num_samples / CONFIG.BATCH_SIZE)

    num_samples = 0
    with open(CONFIG.VAL_PATH, 'r', encoding='utf-8') as fp:
        num_samples = sum(1 for line in fp if "<|endoftext|>" in line)
    VAL_STEPS = math.ceil(num_samples / CONFIG.BATCH_SIZE)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG.EPOCHS,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        callbacks=[earlystop_cb, checkpoint_cb, reduce_lr_cb, csv_logger_cb]
    )

    # Plotting the learning curve of the present model
    plot_learning_curve(history=history, timestamp=timestamp)