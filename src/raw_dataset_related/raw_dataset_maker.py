import polars


def full_raw_dataset_creation():
    train = polars.read_csv('datasets/train-raw.csv')
    test = polars.read_csv('datasets/test-raw.csv')

    dataset = train.vstack(test)

    print(dataset.head())

    dataset.write_csv('datasets/dad-jokes.csv')

    return 'Dataset saved !\n'