# This script downloads the dataset from HuggingFace Datasets.
# The dataset is a 2 column csv file: question, response.
# Originally the dataset is divided into train and test files separately.

from datasets import load_dataset


def download_dataset():
    dataset = load_dataset('shuttie/dadjokes')

    dataset['train'].to_csv('datasets/train-raw.csv')
    dataset['test'].to_csv('datasets/test-raw.csv')

    return 'Dataset downloaded !\n'