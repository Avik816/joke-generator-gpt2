# This scripts adds the separated files into one single csv file,
# as the files are unequally divided. Also, there is no Validation set.
# Custom-defined ratio will be used to split the files into its respective train-val-test sets.

import polars


def full_raw_dataset_creation():
    train = polars.read_csv('datasets/train-raw.csv')
    test = polars.read_csv('datasets/test-raw.csv')

    dataset = train.vstack(test)

    print(dataset.head())

    dataset.write_csv('datasets/dad-jokes.csv')

    return 'Dataset saved !\n'