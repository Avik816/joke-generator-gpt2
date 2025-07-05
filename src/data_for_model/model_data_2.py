'''import tensorflow as tf
from transformers import GPT2Tokenizer
from .. import CONFIG


tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad_token, so use eos_token

def encode_example(line):
    # Tokenize the input text
    encoded = tokenizer(
        line.numpy().decode('utf-8'),
        padding='max_length',
        truncation=True,
        max_length=CONFIG.BLOCK_SIZE,
        return_attention_mask=True
    )

    input_ids = tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(encoded['attention_mask'], dtype=tf.int32)
    labels = tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32)

    return input_ids, attention_mask, labels


def tf_encode_example(line):
    input_ids, attention_mask, labels = tf.py_function(
        func=encode_example,
        inp=[line],
        Tout=(tf.int32, tf.int32, tf.int32)
    )

    # Set shapes manually (important for batching)
    input_ids.set_shape([CONFIG.BLOCK_SIZE])
    attention_mask.set_shape([CONFIG.BLOCK_SIZE])
    labels.set_shape([CONFIG.BLOCK_SIZE])

    # Return as a dictionary (model expects this)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def load_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(tf_encode_example)

    # Define padded shapes for dict elements
    padded_shapes = {
        'input_ids': [CONFIG.BLOCK_SIZE],
        'attention_mask': [CONFIG.BLOCK_SIZE],
        'labels': [CONFIG.BLOCK_SIZE]
    }

    padding_values = {
        'input_ids': tokenizer.pad_token_id,
        'attention_mask': 0,
        'labels': tokenizer.pad_token_id
    }

    dataset = dataset.padded_batch(CONFIG.BATCH_SIZE, padded_shapes=padded_shapes, padding_values=padding_values)
    return dataset

def create_model_dataset():
    train_dataset = load_dataset(CONFIG.TRAIN_PATH)
    val_dataset = load_dataset(CONFIG.VAL_PATH)
    #test_dataset = load_dataset(CONFIG.TEST_PATH)

    return train_dataset, val_dataset'''


import tensorflow as tf
from transformers import GPT2Tokenizer
from .. import CONFIG

# Load tokenizer and define pad token
tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad_token, so we use eos_token

# Function to encode each example
def encode_example(line):
    encoded = tokenizer(
        line.numpy().decode('utf-8'),
        padding='max_length',
        truncation=True,
        max_length=CONFIG.BLOCK_SIZE,
        return_attention_mask=True
    )
    input_ids = tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(encoded['attention_mask'], dtype=tf.int32)
    labels = tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32)
    
    return input_ids, attention_mask, labels

# Wrap encoding function for tf.data API
def tf_encode_example(line):
    input_ids, attention_mask, labels = tf.py_function(
        func=encode_example,
        inp=[line],
        Tout=(tf.int32, tf.int32, tf.int32)
    )
    input_ids.set_shape([CONFIG.BLOCK_SIZE])
    attention_mask.set_shape([CONFIG.BLOCK_SIZE])
    labels.set_shape([CONFIG.BLOCK_SIZE])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Generic loader with shuffle and prefetch
def load_dataset(file_path, shuffle):
    dataset = tf.data.TextLineDataset(file_path)
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=1000)  # ✅ Shuffle for generalization
    dataset = dataset.map(tf_encode_example, num_parallel_calls=tf.data.AUTOTUNE)

    padded_shapes = {
        'input_ids': [CONFIG.BLOCK_SIZE],
        'attention_mask': [CONFIG.BLOCK_SIZE],
        'labels': [CONFIG.BLOCK_SIZE]
    }
    padding_values = {
        'input_ids': tokenizer.pad_token_id,
        'attention_mask': 0,
        'labels': tokenizer.pad_token_id
    }

    dataset = dataset.padded_batch(CONFIG.BATCH_SIZE, padded_shapes=padded_shapes, padding_values=padding_values)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # ✅ Prefetch for efficiency
    
    return dataset

# Main function to return datasets
def create_model_dataset():
    train_dataset = load_dataset(CONFIG.TRAIN_PATH, shuffle=True)
    val_dataset = load_dataset(CONFIG.VAL_PATH, shuffle=False)
    # test_dataset = load_dataset(CONFIG.TEST_PATH)
    
    return train_dataset, val_dataset