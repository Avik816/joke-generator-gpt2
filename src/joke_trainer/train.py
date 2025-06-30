# Model trainng will be done here
# all the necessay items will be imported here

import tensorflow as tf
from .model import model
from .data import train_dataset, val_dataset
from .config import EPOCHS, OUTPUT_DIR
import os

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
    save_best_only=True,
    monitor="val_loss",
    mode="min"
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

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

model.save_pretrained(OUTPUT_DIR)

# after this the plotting function will be called !