from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .. import CONFIG
import tensorflow as tf


def generate_joke():
    # Loading tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.OUTPUT_DIR_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

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

    input_ids = tokenizer.encode("Q: What are a hipster's favorite headphones A:", return_tensors='tf')
                                 
    output = model.generate(
        input_ids,
        max_length=CONFIG.BLOCK_SIZE,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)