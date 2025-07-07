import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .. import CONFIG


def generate_joke():
    # Loading the model
    tf_model = tf.keras.models.load_model('', compile=False) # model path to be added
    hf_model = TFGPT2LMHeadModel.from_pretrained(CONFIG.MODEL_NAME)
    hf_model.set_weights(tf_model.get_weights())

    # Loading the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode('', return_tensors='tf')
                                 
    output = hf_model.generate(
        input_ids,
        max_length=CONFIG.BLOCK_SIZE,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)