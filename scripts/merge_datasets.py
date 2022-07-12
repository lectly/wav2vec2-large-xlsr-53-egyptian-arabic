import pandas as pd
from sklearn.model_selection import train_test_split


def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)


if __name__ == '__main__':
    data_dir = "./data/"

    train = pd.read_csv(f"{data_dir}train.csv")
    dev = pd.read_csv(f"{data_dir}dev.csv")
    test = pd.read_csv(f"{data_dir}test.csv")
    eacsc = pd.read_csv(f"{data_dir}eacsc.csv")
    dataset = pd.concat([train, test, dev, eacsc])
    shuffled_dataset = shuffle_data(dataset)
    train, dev_test = train_test_split(shuffled_dataset, test_size=0.3)
    dev, test = train_test_split(dev_test, test_size=0.5)

    train.to_csv(f"{data_dir}train.csv", index=False)
    dev.to_csv(f"{data_dir}dev.csv", index=False)
    test.to_csv(f"{data_dir}test.csv", index=False)
