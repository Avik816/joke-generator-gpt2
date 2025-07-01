# Model trainng will be done here
# all the necessay items will be imported here

import tensorflow as tf
from .model import model
from data_for_model.model_data_2 import create_model_dataset
from .. import CONFIG
import os


train_dataset, val_dataset = create_model_dataset()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CONFIG.OUTPUT_DIR, 'best_model.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=2,
    restore_best_weights=True
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

def training_gpt2_small():
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG.EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )

    model.save_pretrained(CONFIG.OUTPUT_DIR)

# after this the plotting function will be called !