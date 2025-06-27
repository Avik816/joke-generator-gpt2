import tensorflow as tf
from transformers import GPT2Tokenizer
from .. import CONFIG


tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def encode_line(line):
    tokens = tokenizer.encode(line.numpy().decode('utf-8'))
    tokens = tokens[:CONFIG.BLOCK_SIZE]
    
    return tf.convert_to_tensor(tokens, dtype=tf.int32)

def tf_encode_line(line):
    return tf.py_function(func=encode_line, inp=[line], Tout=tf.int32)

def load_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(tf_encode_line)
    dataset = dataset.padded_batch(
        CONFIG.BATCH_SIZE,
        padded_shapes=[CONFIG.BLOCK_SIZE],
        padding_values=tokenizer.pad_token_id
    )

    return dataset


train_dataset = load_dataset(CONFIG.TRAIN_PATH)
val_dataset = load_dataset(CONFIG.VAL_PATH)
test_dataset = load_dataset(CONFIG.TEST_PATH)