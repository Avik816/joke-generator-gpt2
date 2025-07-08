'''
This script downloads and prepares the model for fine-tuning.

GPT-2(small version) Model is used. This model has 124M+ params.
For fine-tuning the GPT-2 model's top 3 decoder layers and the lm_layer were kept unfrozen (trainable).

By default the model will be saved in the .cache folder (in windows), unless a custom env is set.
'''

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


def setup_model_and_tokenizer(model_name, learning_rate):
    # Loading tokenizer and defining pad token
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad_token, so we use eos_token of GPT2

    # Loading model and setting up with the tokenizer
    gpt2_small_model = TFGPT2LMHeadModel.from_pretrained(model_name)
    gpt2_small_model.resize_token_embeddings(len(tokenizer))
    gpt2_small_model.config.pad_token_id = tokenizer.pad_token_id

    # ======================
    # Freezing all layers first
    # ======================
    for layer in gpt2_small_model.transformer.h:
        layer.trainable = False

    gpt2_small_model.transformer.wte.trainable = False  # word embeddings
    gpt2_small_model.transformer.wpe.trainable = False  # position embeddings

    # ======================
    # Unfreezing last 3 transformer layers
    # ======================
    for layer in gpt2_small_model.transformer.h[-3:]:
        layer.trainable = True

    # ======================
    # Unfreezing final layer norm
    # ======================
    gpt2_small_model.transformer.ln_f.trainable = True

    # ======================
    # Verifying LM head trainability (tied with wte by default)
    print("Trainable variables related to output:")
    for var in gpt2_small_model.trainable_variables:
        if "lm_head" in var.name:
            print("✅", var.name)

    # Compiling model
    gpt2_small_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Model summary
    gpt2_small_model.summary()
    
    '''for layer in gpt2_small_model.layers:
        print(layer.name)'''

    return gpt2_small_model, tokenizer