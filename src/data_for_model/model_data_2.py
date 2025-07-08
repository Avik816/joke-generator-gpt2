'''
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
'''


'''
Preparing the data for the model.
Tokenizing the data and converting it to model specific tensor.
And returning the dictionary that carries this info for the model to ultimately learn.

Shuffling is used for introducing randomization to the model and thus generalizing it better.
A buffer of 1000 is used.
'''

import tensorflow as tf
from .. import CONFIG


def get_encode_example_fn(tokenizer):
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
    return encode_example

def get_tf_encode_example(tokenizer):
    encode_example = get_encode_example_fn(tokenizer)

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

    return tf_encode_example

# Generic loader with shuffle and prefetch
def load_dataset(file_path, shuffle, tokenizer):
    dataset = tf.data.TextLineDataset(file_path)
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=1000)  # ✅ Shuffling for generalization
    
    tf_encode_example = get_tf_encode_example(tokenizer)
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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # ✅ Prefetching the next batch for efficiency
    
    return dataset

# Function to return datasets
def create_model_dataset(tokenizer):
    train_dataset = load_dataset(CONFIG.TRAIN_PATH, True, tokenizer)
    val_dataset = load_dataset(CONFIG.VAL_PATH, False, tokenizer)
    # test_dataset = load_dataset(CONFIG.TEST_PATH)
    
    return train_dataset, val_dataset