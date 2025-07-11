from src import (
    download_dataset,
    full_raw_dataset_creation,
    cleaning_dataset,
    data_splitting,
    training_gpt2_small,
    model_testing_evaluation,
    generate_joke
)
import os
import warnings
import logging

# Setting environment variables to suppress TensorFlow C++ logs
# '2' means suppress all INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Explicitly setting TF_ENABLE_ONEDNN_OPTS=0 to silence that specific message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Using a blanket filter to ignore all warnings from all libraries
warnings.filterwarnings("ignore")

# Setting logging levels for specific libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

"""print('\nDOWNLOADING DATASET from HuggingFace ... ')
download_dataset()

print('\n\nCREATING FULL RAW DATASET ...')
full_raw_dataset_creation()

print('\n\nPREPROCESSING THE DATASET ...')
cleaning_dataset()

print('\n\nSPLITTING DATASET into TRAIN-VAL-TEST SETS ...')
data_splitting()

print('\n\nMODEL TRAINING ...')
training_gpt2_small()"""

print('\n\nMODEL TESTING and EVALUATION ...')
print(model_testing_evaluation())

print('\n\nSAMPLE JOKE GENERATION ...')
print(generate_joke())