# Main entry point of the system.

from src import *
import warnings
import transformers

# Set the verbosity to 'error', which will suppress warnings and info messages
transformers.logging.set_verbosity_error()

warnings.filterwarnings('ignore')

"""print('\nDownloading Dataset from HuggingFace ... ')
download_dataset()

print('\n\nCreating the raw full dataset ...')
full_raw_dataset_creation()

print('\n\nPreprocessing the dataset ...')
cleaning_dataset()

print('\n\nSplitting dataset into Train-Val-Test ...')
data_splitting()

print('\n\nModel Training ...')
training_gpt2_small()

print('\n\nModel Testing and Evaluation ...')
print(model_testing_evaluation())"""

print('\n\nSample joke generation ...')
print(generate_joke())