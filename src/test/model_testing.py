import tensorflow as tf
from .. import CONFIG
from ..data_for_model.model_data_2 import load_dataset
import math


# Loading model and tokenizer
model = tf.keras.models.load_model() # Model path to be added later

# Test Set data
test_set = load_dataset(CONFIG.TEST_PATH, False)

# Evaluate on test set
eval_loss = model.evaluate(test_set)
print(f'\nTest loss: {eval_loss:.4f}')

# Perplexity Score: Describes how the model predicts the next token
print(f'The Perplexity Score: {math.exp(eval_loss)}')