import tensorflow as tf
from .. import CONFIG
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from ..data_for_model.model_data_2 import load_dataset
import math


def model_testing_evaluation():
    # Loading tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.OUTPUT_DIR_TOKENIZER)
    # tokenizer.pad_token = tokenizer.eos_token

    # Loading model
    model = TFGPT2LMHeadModel.from_pretrained(CONFIG.SAVED_MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Compiling the model
    learning_rate = CONFIG.BEST_LR # to be added after training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    model.load_weights('model_weights/model_weights_20250711-062654_epoch-01_val_loss-0.033497.h5')

    # Test Set data
    test_set = load_dataset(CONFIG.TEST_PATH, False, tokenizer)

    # Evaluate on test set
    eval_loss = model.evaluate(test_set, verbose=1)
    print(f'\nTest loss: {eval_loss:.6f}')

    # Perplexity Score: Describes how the model predicts the next token
    print(f'The Perplexity Score: {math.exp(eval_loss)}')