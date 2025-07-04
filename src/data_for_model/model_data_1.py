import polars
from .. import CONFIG


def write_into_file(dataset, filename):
    sample_list = [samples + '\n' for samples in dataset['jokes']]

    with open(file=f'data/jokes-{filename}.txt', mode='w', encoding='UTF-8') as fp:
        fp.writelines(sample_list)

def data_splitting():
    dataset = polars.read_csv('datasets/formatted-dad-jokes.csv')

    train = dataset[:int(dataset.shape[0] * CONFIG.TRAIN_SIZE), :]
    val = dataset[int(train.shape[0]):int(train.shape[0] + int(dataset.shape[0] * CONFIG.VAL_SIZE)), :]
    test = dataset[int(val.shape[0]):int(val.shape[0] + int(dataset.shape[0] * CONFIG.TEST_SIZE)), :]

    write_into_file(train, 'train')
    print('Training file saved in data/ !')

    write_into_file(val, 'val')
    print('Validation file saved in data/ !')
    
    write_into_file(test, 'test')
    print('Test file saved in data/ !')