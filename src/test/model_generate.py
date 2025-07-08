import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .. import CONFIG


def generate_joke():
    # Loading the model
    # need to load the weights and update the weights to the model
    model = TFGPT2LMHeadModel.from_pretrained(CONFIG.MODEL_NAME)
    model.load_weights() # path to best model weights

    # Loading the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode('Q: ', return_tensors='tf')
                                 
    output = model.generate(
        input_ids,
        max_length=CONFIG.BLOCK_SIZE,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)