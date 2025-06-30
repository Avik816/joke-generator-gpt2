from transformers import TFGPT2LMHeadModel
import tensorflow as tf
from .. import CONFIG


# Need to load the tokenizer here
# make itr all into a function

gpt2_small_model = TFGPT2LMHeadModel.from_pretrained(CONFIG.MODEL_NAME)
gpt2_small_model.resize_token_embeddings(len(tokenizer))
gpt2_small_model.config.pad_token_id = tokenizer.pad_token_id

# Freeze all transformer blocks
for layer in gpt2_small_model.transformer.h:
    layer.trainable = False

# Freeze embeddings
gpt2_small_model.transformer.wte.trainable = False
gpt2_small_model.transformer.wpe.trainable = False

# Unfreeze last 3 transformer layers
for layer in gpt2_small_model.transformer.h[-3:]:
    layer.trainable = True

# Unfreeze final layer norm and lm_head
gpt2_small_model.transformer.ln_f.trainable = True
gpt2_small_model.lm_head.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

gpt2_small_model.compile(optimizer=optimizer, loss=loss_fn)

gpt2_small_model.summary()