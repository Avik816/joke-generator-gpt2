from datasets import load_dataset

def download_dataset():
    dataset = load_dataset('shuttie/dadjokes')

    dataset['train'].to_csv('datasets/train-raw.csv')
    dataset['test'].to_csv('datasets/test-raw.csv')

    return 'Dataset downloaded !\n'