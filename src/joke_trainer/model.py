from ..data_for_model.model_data_2 import tokenizer
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


def setup_model(model_name, learning_rate):
    # Load model
    gpt2_small_model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_small_model.resize_token_embeddings(len(tokenizer))
    gpt2_small_model.config.pad_token_id = tokenizer.pad_token_id

    # ======================
    # Freeze all layers first
    # ======================
    for layer in gpt2_small_model.transformer.h:
        layer.trainable = False

    gpt2_small_model.transformer.wte.trainable = False  # word embeddings
    gpt2_small_model.transformer.wpe.trainable = False  # position embeddings

    # ======================
    # Unfreeze last 3 transformer layers
    # ======================
    for layer in gpt2_small_model.transformer.h[-3:]:
        layer.trainable = True

    # ======================
    # Unfreeze final layer norm
    # ======================
    gpt2_small_model.transformer.ln_f.trainable = True

    # ======================
    # Verify LM head trainability (tied with wte by default)
    print("Trainable variables related to output:")
    for var in gpt2_small_model.trainable_variables:
        if "lm_head" in var.name:
            print("✅", var.name)

    # Compile model
    gpt2_small_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Print model summary
    gpt2_small_model.summary()
    
    for layer in gpt2_small_model.layers:
        print(layer.name)

    return gpt2_small_model