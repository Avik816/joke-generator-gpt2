# Dataset sizes
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

# Model parameters
MODEL_NAME = 'gpt2'
TRAIN_PATH = 'data/jokes-train.txt'
VAL_PATH = 'data/jokes-val.txt'
TEST_PATH = 'data/jokes-test.txt'
OUTPUT_DIR_MODEL_WEIGHTS = 'model_weights'
OUTPUT_DIR_MODEL = 'saved_model'
OUTPUT_DIR_TOKENIZER = 'tokenizer'
OUTPUT_DIR_PLOT = 'plots'
BEST_MODEL_PATH = 'models/best_model_20250707-000312_epoch-13_loss-0.0008.keras'

# Model hyper-parameters
BATCH_SIZE = 4
EPOCHS = 100
BLOCK_SIZE = 50
LEARNING_RATE = 5e-5
BEST_LR = '' # to be added after training