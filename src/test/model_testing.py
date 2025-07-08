import tensorflow as tf
from .. import CONFIG
from transformers import TFGPT2LMHeadModel
from ..data_for_model.model_data_2 import load_dataset
import math


def model_testing_evaluation():
    # Loading model
    model = TFGPT2LMHeadModel.from_pretrained(CONFIG.MODEL_NAME)
    model.load_weights() # path to best model weights

    # Compiling the model
    learning_rate = CONFIG.BEST_LR # to be added after training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Test Set data
    test_set = load_dataset(CONFIG.TEST_PATH, False)

    # Evaluate on test set
    eval_loss = model.evaluate(test_set, verbose=1)
    print(f'\nTest loss: {eval_loss:.6f}')

    # Perplexity Score: Describes how the model predicts the next token
    print(f'The Perplexity Score: {math.exp(eval_loss)}')