import pandas as pd
from os.path import exists


if __name__ == '__main__':
    data_dir = "./data/"
    wav_dir = "./mgb3/wav/"
    files = ["train", "test", "dev"]
    for file in files:
        df = pd.read_csv(f"{data_dir}{file}.csv")
        for index, row in df.iterrows():
            path = wav_dir + row["filename"] + ".wav"
            if not exists(path):
                df.drop([index], inplace=True)
        df.sort_values(by="filename", inplace=True)
        df.to_csv(f"{data_dir}{file}.csv", index=False)
